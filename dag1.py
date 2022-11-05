'''
Author: RenzeLou marionojump0722@gmail.com
Date: 2022-09-14 22:11:12
LastEditors: RenzeLou marionojump0722@gmail.com
LastEditTime: 2022-11-05 23:57:00
FilePath: /7-bare-RUN/dag1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausePredictor,PaG
from encoder import UtterEncoder2


class CauseDagNoEmotion(nn.Module):
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
                 num_bases
                        ):
        super(CauseDagNoEmotion, self).__init__()
        self.utter_encoder = UtterEncoder2(utter_dim, emo_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead)
        self.pag = PaG(window,utter_dim,num_bases)
        self.classifier = CausePredictor(utter_dim, utter_dim)

    def forward(self, input_ids, attention_mask, mask, adj,label):
        utter_emb = self.utter_encoder(input_ids, attention_mask,adj,label) # [batch_size, conv_size, utter_dim]
        utter_emb,rel_emb_k,rel_emb_v = self.pag(utter_emb)
        logits = self.classifier(utter_emb,rel_emb_k,rel_emb_v,mask)
        
        return logits
