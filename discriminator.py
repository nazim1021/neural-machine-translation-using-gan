import torch
import torch.nn as nn
import torch.nn.functional as F 


class Discriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.fixed_max_len = args.fixed_max_len
        self.use_cuda = use_cuda


        self.embed_src_tokens = Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())


        self.conv1 = nn.Sequential(
            Conv2d(in_channels=2000,
                   out_channels=512,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=512,
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 12 * 12, 20),
            nn.ReLU(),
            nn.Dropout(),
            Linear(20, 20),
            nn.ReLU(),
            Linear(20, 1),
        )

    def forward(self, src_sentence, trg_sentence):
        batch_size = src_sentence.size(0)

        src_out = self.embed_src_tokens(src_sentence)
        trg_out = self.embed_src_tokens(trg_sentence)

        src_out = torch.stack([src_out] * trg_out.size(1), dim=2)
        trg_out = torch.stack([trg_out] * src_out.size(1), dim=1)
        
        out = torch.cat([src_out, trg_out], dim=3)
        
        out = out.permute(0,3,1,2)
        
        out = self.conv1(out)
        out = self.conv2(out)
        
        out = out.permute(0, 2, 3, 1)
        
        out = out.contiguous().view(batch_size, -1)
        
        out = torch.sigmoid(self.classifier(out))

        return out

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m