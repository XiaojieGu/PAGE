
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from model import CausePredictor
import numpy as np
import torch.nn.init as init



#Multi_att 0.1
#GAT
#MLP = 0.1
#Tranms


    
class UtterEncoder2(nn.Module):
    def __init__(self, model_size, utter_dim, conv_encoder='none', rnn_dropout=None,emotion_emb=None,emotion_dim=200):
        super(UtterEncoder2, self).__init__()
        encoder_path = 'roberta-' + model_size
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        if model_size == 'base':
            token_dim = 768
        else:
            token_dim = 1024
        #两处
        self.mapping = nn.Linear(token_dim, utter_dim)
        self.dag = DAGNN(utter_dim, utter_dim, num_layers=1, dropout=0.0, pooler_type='all')

        
        self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
        self.emotion_lin = nn.Linear(emotion_emb.shape[1], 200)
        self.emotion_mapping = nn.Linear(300 + 200, utter_dim)

        
    def forward(self, conv_utterance, attention_mask, conv_len,know_input_ids,know_attention_mask,adj,edge_mask,s_mask,o_mask,emotion_label):
        # conv_utterance: [[conv_len1, max_len1], [conv_len2, max_len2], ..., [conv_lenB, max_lenB]]
        processed_output = []
        
        for cutt, amsk,know,k_amsk in zip(conv_utterance, attention_mask,know_input_ids,know_attention_mask):
            #用包含了整个句子的聚合表示cls_head来表示整个句子的表征
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state  #torch.Size([12, 15, 768])
            
            # print(output_data.device)
            #最大池化，从word-level到clause-level表征
            pooler_output = torch.max(output_data, dim=1)[0]  #torch.Size([6, 768])
  
            mapped_output = self.mapping(pooler_output)  #torch.Size([12, 300])
          
            
            processed_output.append(mapped_output)

        conv_output = pad_sequence(processed_output, batch_first=True)   

        emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
      
        
        utter_emb = self.emotion_mapping(torch.cat([conv_output, emo_emb], dim=-1))

        utter_emb = self.dag(utter_emb,adj)  ##torch.Size([8, 11, 300])

        return utter_emb  # [batch_size, conv_size, utter_dim]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, emb_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        # self.v_proj_weight_s = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        # self.v_proj_weight_o = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.o_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.know_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.dropout = 0.1
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        # torch.nn.init.xavier_uniform_(self.v_proj_weight_s)
        # torch.nn.init.xavier_uniform_(self.v_proj_weight_o)
        torch.nn.init.xavier_uniform_(self.v_proj_weight)
        torch.nn.init.xavier_uniform_(self.o_proj_weight)
        torch.nn.init.xavier_uniform_(self.know_proj_weight)

    def forward(self, x, adj):
        # knowledge: (knum, nh*hdim), know_adj: (bsz, slen, slen)
        # adj, s_mask, o_mask: (bsz, slen, slen)
        # input size: (slen, bsz, nh*hdim)
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(self.dropout(linear_out) + x)
        return output

class TransformerLayer(nn.Module):
        def __init__(self, emb_dim, nhead, ff_dim, att_dropout, dropout):
            super(TransformerLayer, self).__init__()
            self.attention = MultiHeadAttention(nhead, emb_dim, att_dropout)

            self.norm = nn.LayerNorm(emb_dim)
            self.dropout = nn.Dropout(dropout)

            self.ff_net = MLP(emb_dim, ff_dim, dropout)

        def forward(self, x, adj):
    
            x2 = self.attention(x, adj)
            ss = x.transpose(0,1) + self.dropout(x2)
            ss = self.norm(ss)
            ff_out = self.ff_net(ss)
            # print(ff_out.shape)  #torch.Size([4, 19, 300])
           
            
            return ff_out


         
class DAGNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, pooler_type='all'):
        super(DAGNN, self).__init__()

        self.trans_layers = nn.ModuleList()
        for i in range(num_layers):
            trans_layer = TransformerLayer(emb_dim=300, nhead=6, ff_dim=128, att_dropout=0.0, dropout=0.0)
            self.trans_layers.append(trans_layer)
        


    def forward(self, features,  adj_mask):
 
        x = self.trans_layers[0](features.transpose(0,1),adj_mask)     
        
        return x



def mask_logic(alpha, adj):
    return alpha - (1 - adj) * 1e30

