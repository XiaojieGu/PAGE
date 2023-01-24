import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Causal_Classifier,PaG
from encoder import UtterEncoder


class PAGE(nn.Module):
    def __init__(self,
                 utter_dim,
                 emo_emb,
                 emotion_dim,
                 att_dropout,
                 mlp_dropout,
                 pag_dropout,
                 ff_dim,
                 nhead,
                 window,
                 num_bases,
                 max_len,
                 posi_dim
                        ):
        super(PAGE, self).__init__()
        self.utter_encoder = UtterEncoder(utter_dim, emo_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead)
        self.pag = PaG(window,utter_dim,num_bases,max_len,posi_dim)
        self.classifier = Causal_Classifier(utter_dim, utter_dim)

    def forward(self, input_ids, attention_mask, mask, adj,label):
        utter_emb = self.utter_encoder(input_ids, attention_mask,adj,label) # [batch_size, conv_size, utter_dim]
        utter_emb,rel_emb_k,rel_emb_v = self.pag(utter_emb)
        logits = self.classifier(utter_emb,rel_emb_k,rel_emb_v,mask)
        
        return logits
