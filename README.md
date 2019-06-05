## Neural Machine Translation using Adversarial Training

Adversarial Training or Generative Adversarial Networks have shown great promise in Image Generation and are able to achieve state-of-the-art results. However, For sequential data such as text, training GANs has proven to be difficult. One reason is because of non-differentiable nature of generating text with R NNs. Consequently, there have been past work where GANs have been employed for NLP tasks such as text generation, sequence labelling, etc. As a part of directed research, I examine one such application of GAN with RNN called Adversarial Neural Machine Translation.

Here, I manage to investigate how NMT-GAN works and implement an NMT-GAN model according to the work of (Lijun Wu et al., 2017) and finally compare it with traditional NMT models. The model was run on german-english and czech-english dataset.

### Dataset

Here, i make use of freely available IWSLT'14 dataset. The dataset is downloaded and preprocessed through facebook `fairseq` toolkit

Follow below steps to download & preprocess the dataset:
1. git clone https://github.com/pytorch/fairseq
2. cd examples/translation/; bash prepare-iwslt14.sh; cd ../.. (make sure to make relevant dataset name changes in bash script)
3. TEXT=examples/translation/iwslt14.tokenized.de-en
4. python preprocess.py --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en

### Usage

The main program for the NMT-GAN is `joint_train.py` .

The file `train_generator.py` is a traditional NMT, which is similar to NMT-GAN generator network.

To train NMT-GAN model, please use following command:
```
python joint_train.py --data data-bin/iwslt15.tokenized.cs-en/  --src_lang de --trg_lang en --learning_rate 1e-3 --joint-batch-size 64 --gpuid 0 --clip-norm 1.0 --epochs 10
```
This will save the model in checkpoints folder (make sure GPU is enabled)

Generate predictions as per below:
```
python generate.py --data data-bin/iwslt14.tokenized.de-en/ --src_lang de --trg_lang en --batch-size 64 --gpuid 0
```

This generates `predictions.txt` and `real.txt` 

As common in most NMT models, we use BLEU score as an evaluation metric. I make use of freely available `mosesedecoder` toolkit.

Follow below steps for evaluation:
1. Postprocess both real and predictions text files as below
```
bash postprocess.sh < real.txt > real_processed.txt
bash postprocess.sh < predictions.txt > predictions_processed.txt
```
2. Run BLEU evaluation
```
perl scripts/multi-bleu.perl real_processed.txt < predictions_processed.txt
```

### References
   1. Code adapted from: https://github.com/wangyirui/Adversarial-NMT
   2. https://github.com/pytorch/fairseq
   3. https://github.com/moses-smt/mosesdecoder/tree/master/scripts
   4. Lijun Wu, Yingce Xia, Li Zhao, Fei Tian, Tao Qin, Jianhuang Lai, and Tie-Yun Liu. Adversarial neural machine translation. arXiv, 2017