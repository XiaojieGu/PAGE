import torch
import torch.nn as nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


class RelativePositionEncoding(nn.Module):
    def __init__(self, input_dim, max_len=10):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        self.pe_k = nn.Embedding(max_len+1, input_dim, padding_idx=0)
        self.pe_v = nn.Embedding(max_len+1, input_dim, padding_idx=0)

    def forward(self, position_mask):
        position_mask = torch.clamp(position_mask, min=0, max=self.max_len).long()
        pemb_k = self.pe_k(position_mask)
        pemb_v = self.pe_v(position_mask)
        
        return pemb_k, pemb_v


def rel_adj_create(rel_adj,slen,window):
    for i in range(slen):
        for s in range(i+1,slen):
            rel_adj[i][s] = 1
    
    for i in range(slen):
        num = 1     
        for o in range(i-1,-1,-2):
            if((o-1)<0):
                rel_adj[i][o] = -num
            else:
                rel_adj[i][o] = -num
                rel_adj[i][o-1] = -num
            num+=1
    
    for i in range(slen):
        for o in range(i-1,-1,-1):
            if(rel_adj[i][o]<-(window+1)):
                rel_adj[i][o] = - (window + 1) 
    
    return rel_adj

def index_create(slen):
    index = []
    start = []
    end = []     
    
    for i in range(0,slen):
        for j in range(0,slen):
            start.append(i)
    for i in range(0,slen):
        for j in range(0,slen):
            end.append(j)

    index.append(start)
    index.append(end)
    
    index = torch.tensor(index).long()
    
    return index

class PaG(nn.Module):
    def __init__(self,window,utter_dim,num_bases):
        super(PaG, self).__init__()
        self.posi = RelativePositionEncoding(100,10)
        self.window = window
        self.rel_num = self.window + 2
        self.rgcn = RGCNConv(utter_dim,utter_dim,self.rel_num,num_bases=num_bases)

    
    def forward(self,x):
        batch_size = x.shape[0]
        x_dim = x.shape[2]
        slen = x.shape[1]
        src_pos = torch.arange(slen).unsqueeze(0)
        tgt_pos = torch.arange(slen).unsqueeze(1)
        pos_mask = (tgt_pos - src_pos) + 1
        pos_mask = pos_mask.to(x.device)
        
        rel_emb_k, rel_emb_v = self.posi(pos_mask)
        
        rel_emb_k = rel_emb_k.unsqueeze(0).expand(batch_size, slen, slen, 100)
        rel_emb_v = rel_emb_v.unsqueeze(0).expand(batch_size, slen, slen, 100)

        rel_adj = (src_pos - tgt_pos).to(x.device)
        
        self.rgcn.to(x.device)

        rel_adj = rel_adj_create(rel_adj,slen,self.window)
        index = index_create(slen).to(x.device)
        
        edge_type = torch.flatten(rel_adj).long().to(x.device)

        out = self.rgcn(x[0],index,edge_type).unsqueeze(0)
        for i in range(1,batch_size):
            h = self.rgcn(x[i],index,edge_type)
            out = torch.cat((out,h.unsqueeze(0)),dim=0)
            
        return out,rel_emb_k,rel_emb_v
    
class CausePredictor(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(CausePredictor, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.posi = RelativePositionEncoding(100,10)
        self.mlp_dropout = mlp_dropout

        self.mlp = nn.Sequential(nn.Linear(2*input_dim+200, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False)

    def forward(self, x,rel_emb_k,rel_emb_v,mask):

        batch_size = x.shape[0]

        x_dim = x.shape[2]
        
        slen = x.shape[1]

        x_source = x.unsqueeze(1).expand(batch_size, slen, slen, x_dim)
        x_target = x.unsqueeze(2).expand(batch_size, slen, slen, x_dim)
        # [batch_size, conv_len, conv_len, 2*x_dim]

        x_source = torch.cat([x_source,rel_emb_k],dim=-1)
        x_target = torch.cat([x_target,rel_emb_v],dim=-1)
        
        x_cat = torch.cat([x_source, x_target], dim=-1)  #torch.Size([8, 21, 21, 600])

        # [batch_size, conv_len, conv_len]
        predict_score = self.predictor_weight(self.mlp(x_cat)).squeeze(-1)
        predict_score = torch.sigmoid(predict_score) * mask
       
        return predict_score



 



    

