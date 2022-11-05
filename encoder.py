
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence



   
class UtterEncoder2(nn.Module):
    def __init__(self, utter_dim, emotion_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead):
        super(UtterEncoder2, self).__init__()
        encoder_path = 'roberta-base'
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        token_dim = 768
        
        self.mapping = nn.Linear(token_dim, utter_dim)

        self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
        self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
        self.emotion_mapping = nn.Linear(utter_dim + emotion_dim, utter_dim)
        
        self.attention = MultiHeadAttention(nhead, utter_dim, att_dropout)
        self.norm = nn.LayerNorm(utter_dim)
        self.dropout = nn.Dropout(pag_dropout)
        self.mlp = MLP(utter_dim, ff_dim, mlp_dropout)

        
    def forward(self, conv_utterance, attention_mask,adj,emotion_label):
        processed_output = []
        for cutt, amsk in zip(conv_utterance, attention_mask):
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state  #torch.Size([12, 15, 768])
            #最大池化，从word-level到clause-level表征
            pooler_output = torch.max(output_data, dim=1)[0]  #torch.Size([6, 768])
            mapped_output = self.mapping(pooler_output)  #torch.Size([12, 300])
            processed_output.append(mapped_output)

        conv_output = pad_sequence(processed_output, batch_first=True)   
        emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
        utter_emb = self.emotion_mapping(torch.cat([conv_output, emo_emb], dim=-1))
        
        x = utter_emb.transpose(0,1)
        x2 = self.attention(x, adj)
        ss = x.transpose(0,1) + self.dropout(x2)
        ss = self.norm(ss)
        out = self.mlp(ss)

        return out  # [batch_size, conv_size, utter_dim]


   
class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, utter_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = utter_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.o_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        torch.nn.init.xavier_uniform_(self.v_proj_weight)
        torch.nn.init.xavier_uniform_(self.o_proj_weight)

    def forward(self, x, adj):

        slen = x.size(0)
        bsz = x.size(1)
        adj = adj.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        adj = adj.contiguous().view(bsz*self.nhead, slen, slen) 
        scaling = float(self.head_dim) ** -0.5
  
        query = x
        key = x
        value = x
        
        # (slen, bsz, nh*hdim) -> (slen, bsz*nh, hdim) -> (bsz*nh, slen, slen, hdim)
        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(2)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        value = value.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        
        attention_weight = query*key
        attention_weight = attention_weight.sum(3) * scaling
      
        attention_weight = mask_logic(attention_weight, adj)
        attention_weight = F.softmax(attention_weight, dim=2)

        attention_weight = F.dropout(attention_weight, p=self.dropout, training=True)

        attn_sum = (value * attention_weight.unsqueeze(3)).sum(2)
        attn_sum = attn_sum.transpose(0, 1).contiguous().view(bsz, slen, -1)
        output = F.linear(attn_sum, self.o_proj_weight)
       
        return output

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(self.dropout(linear_out) + x)
        return output


def mask_logic(alpha, adj):
    return alpha - (1 - adj) * 1e30

