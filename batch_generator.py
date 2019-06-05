'''

This code is adapted from Facebook Fairseq-py
Visit https://github.com/facebookresearch/fairseq-py for more information

'''

import math
import torch
import torch.nn.functional as F

class BatchGenerator(object):
    def __init__(self, model, beam_size=1, minlen=1, maxlen=None,
                 stop_early=True, normalize_scores=True, len_penalty=1,
                 unk_penalty=0):
        """Generates translations of a given source sentence.

        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.model = model
        self.pad = model.dst_dict.pad()
        self.unk = model.dst_dict.unk()
        self.eos = model.dst_dict.eos()

        self.vocab_size = len(model.dst_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = model.decoder.max_positions()
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty

        # use for debug
        self.trauncate_cnt = 0

    def cuda(self):
        self.model.cuda()
        return self

    def generate_translation_tokens(self, args, dataset, sample, beam_size=None, maxlen_a=0.0, maxlen_b=None, nbest=1):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        input = sample['net_input']
        srclen = input['src_tokens'].size(1)
        # with torch.no_grad():
        hypos = self.generate(
            input['src_tokens'],
            input['src_lengths'],
            beam_size=beam_size,
            maxlen=int(maxlen_a*srclen + maxlen_b)
        )
        # set the tensor has the same width as max possible translation
        max_res = int(maxlen_a*srclen + maxlen_b)
        pred_tokens = input['src_tokens'].new(len(hypos), max_res).fill_(self.pad)
        for i, hypo in enumerate(hypos): # batch traverse
            hypo_tokens = hypo[:min(len(hypo), nbest)][0]['tokens']
            # hypo_str = dataset.dst_dict.string(hypo_tokens, args.remove_bpe)
            # trg_str = dataset.dst_dict.string(sample['target'][i], args.remove_bpe)
            # print("Trans: {0}".format(hypo_str))
            # print("Target: {0}".format(trg_str))

            # truncate the prediction if exceeds the maxlen
            if hypo_tokens.size(0) > max_res:
                hypo_tokens = hypo_tokens[:max_res]
            pred_tokens[i,:hypo_tokens.size(0)] = hypo_tokens

        return pred_tokens


    def generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None):
        """Generate a batch of translations."""
        with torch.no_grad():
            return self._generate(src_tokens, src_lengths, beam_size, maxlen)

    def _generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None):
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)

        incremental_states = {}
        model = self.model
        model.eval()
        incremental_states[model] = {}

        # compute the encoder output for each beam
        encoder_out = model.encoder(
            src_tokens.repeat(1, beam_size).view(-1, srclen),
            src_lengths.repeat(beam_size),
        )

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
        attn_buf = attn.clone()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz)*beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}
        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step+2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2]

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step+1)**self.len_penalty

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                sent = idx // beam_size
                sents_seen.add(sent)

                def get_hypo():
                    _, alignment = attn_clone[i].max(dim=0)
                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': attn_clone[i],  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            # return number of hypotheses finished this step
            num_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        reorder_state = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                model.decoder.reorder_incremental_state(
                    incremental_states[model], reorder_state)

            probs, avg_attn_scores = self._decode(
                tokens[:, :step+1], encoder_out, incremental_states)
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
                scores = scores.type_as(probs)
                scores_buf = scores_buf.type_as(probs)
            else:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores[:, step-1].view(-1, 1))
            probs[:, self.pad] = -math.inf  # never select pad
            probs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            attn[:, :, step+1].copy_(avg_attn_scores)

            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                # take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                torch.topk(
                    probs.view(bsz, -1),
                    k=min(cand_size, probs.view(bsz, -1).size(1) - 1),  # -1 so we never select pad
                    out=(cand_scores, cand_indices),
                )
                torch.div(cand_indices, self.vocab_size, out=cand_beams)
                cand_indices.fmod_(self.vocab_size)
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    probs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= finalize_hypos(
                    step, eos_bbsz_idx, eos_scores)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    num_remaining_sent -= finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets)*cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step+1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step+1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step+1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            torch.index_select(
                attn[:, :, :step+2], dim=0, index=active_bbsz_idx,
                out=attn_buf[:, :, :step+2],
            )

            # swap buffers
            old_tokens = tokens
            tokens = tokens_buf
            tokens_buf = old_tokens
            old_scores = scores
            scores = scores_buf
            scores_buf = old_scores
            old_attn = attn
            attn = attn_buf
            attn_buf = old_attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(bsz):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_out, incremental_states):

        with torch.no_grad():
            decoder_out, attn = self.model.decoder(tokens, encoder_out, incremental_states[self.model])
        probs = F.log_softmax(decoder_out[:, -1, :], dim=1)

        if attn is not None:
            attn = attn[:, -1, :].data

        return probs, attn
