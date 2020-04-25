import torch
import torch.nn as nn

import pickle

from Sublayers import FeedForward, MultiHeadAttention, Norm

def get_one_hot_vectors(input_sequence, y_t):
  """This function takes in input sequence and target at time step t
  and returns a one hot vector (named alpha_hat_t_i in Eq. 3 of Song et al. paper)

  Arguments:
      input_sequence {torch tensor} -- contains tokens of input sequence
      y_t {int} -- token of target at time step t

  Returns:
      one_hot_vector {torch tensor}
  """    
  dictionary = pickle.load(open('data/tokenized_translation_dictionary.p', 'rb'))
  one_hot_vector = torch.zeros(len(input_sequence))
  for i, en_word in enumerate(input_sequence):
      if dictionary.get(en_word, None) == y_t:
          one_hot_vector[i] = 1
  return one_hot_vector


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask, src_tokens, target_token):
        # decoder attention block
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)

        # beta's
        matrix_plot = torch.bmm(e_outputs, torch.transpose(x2, 1, 2))
        betas = matrix_plot[:, -1]

        print(src_tokens, target_token)
        alphas = get_one_hot_vectors(src_tokens, target_token)
        print(len(alphas), len(betas))
        assert len(betas) == len(alphas)

        # encoder-decoder attention block
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x