import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask, src_tokens, target_token):
        x = self.embed(trg)
        x = self.pe(x)
        align_loss = 0
        for i in range(self.N):
            x, block_align_loss = self.layers[i](x, e_outputs, src_mask, trg_mask, src_tokens=src_tokens, target_token=target_token)
            align_loss += block_align_loss
        return self.norm(x), align_loss / self.N

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output, align_loss = self.decoder(trg, e_outputs, src_mask, trg_mask, src_tokens=src, target_token=trg)
        output = self.out(d_output)
        return output, align_loss

def get_model(opt, src_vocab, trg_vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    model = model.cuda()


    dictionary = pickle.load(open('data/tokenized_translation_dictionary.p', 'rb'))
    max_row = max(max(dictionary.keys()), src_vocab)+1
    max_col= max(max(dictionary.values()), trg_vocab)+1
    EF = torch.zeros(max_row, max_col).cuda()
    for k, v in dictionary.items():
        EF[k, v] = 1

    pickle(EF, open('data/EF.p', 'wb'))
    
    return model
    
