'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-13 23:02:09
LastEditors: RenzeLou marionojump0722@gmail.com
LastEditTime: 2022-10-17 01:16:00
FilePath: /bare-RUN/dag1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausePredictor
from encoder import UtterEncoder2
from torchinfo import summary

#共三处dropout
class CauseDagNoEmotion(nn.Module):
    def __init__(self,
                dep_rel_num,
                 model_size,
                 utter_dim,
                 dep_relation_embed_dim,
                 mem_dim,
                 conv_encoder,
                 rnn_dropout,
                 num_layers,
                 gcn_dropout,
                 emo_emb,
                 emotion_dim
                        ):
        super(CauseDagNoEmotion, self).__init__()
        # self.utter_encoder = UtterEncoder(model_size, mapping_type, utter_dim, conv_encoder, rnn_dropout)
        self.utter_encoder = UtterEncoder2(model_size, utter_dim, conv_encoder, rnn_dropout,emo_emb,emotion_dim)
        # self.dag = DAGNN(utter_dim, utter_dim, num_layers, dropout, pooler_type)
        #一处
        # self.mixed_gat = Mixed_GAT(utter_dim, dep_rel_num, mem_dim=300, dep_relation_embed_dim=300,hidden_size=64, gcn_dropout=0.1, num_layers=2)
        self.classifier = CausePredictor(utter_dim, utter_dim)

    def forward(self, input_ids, attention_mask, conv_len, mask, adj,rel_adj,know_input_ids,know_attention_mask,edge_mask,s_mask,o_mask,label):

        utter_emb = self.utter_encoder(input_ids, attention_mask, conv_len,know_input_ids,know_attention_mask,adj,edge_mask,s_mask,o_mask,label) # [batch_size, conv_size, utter_dim]

        # utter_emb = self.dag(utter_emb, e_mask, s_mask, o_mask)
        # utter_emb = self.mixed_gat(edge_mask,rel_adj,utter_emb)
        
        # net = self.classifier(utter_emb, mask)
        # total = sum([param.numel() for param in net.parameters()])
        # print("total model param num:",total)
        # sum = summary(self.classifier, input_size=(utter_emb, mask))
        # print(sum)
        logits = self.classifier(utter_emb, mask)
        
        return logits
