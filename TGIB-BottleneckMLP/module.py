import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)
    
class MergeLayer_with_fc(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5, dim6, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.fc3 = nn.Linear(dim4, dim5)
        self.fc4 = nn.Linear(dim5, dim6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.dropout(self.act(self.fc1(x)))
        h = self.dropout(self.act(self.fc2(h)))
        h = self.dropout(self.act(self.fc3(h)))
        return self.fc4(h)




class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        # print('q', q.shape)
        # print('k', k.shape)
        # print('v', v.shape)
        # print(self.w_ks)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()) 
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        # pdb.set_trace()
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1] 
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim] # [200, 1, 1] * [1, 1, 172] = [200, 1, 172]
        map_ts += self.phase.view(1, 1, -1) # + [1, 1, 172]
        
        harmonic = torch.cos(map_ts) # cosine

        return harmonic #self.dense(harmonic)
    
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None
    

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        #print('q shape here 1', q.shape)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]


        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # pdb.set_trace()

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze(dim=1)
        attn = attn.squeeze(dim=1)

        # pdb.set_trace()
        output = self.merger(output, src)
        return output, attn

class TGIB(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, hidden,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        super(TGIB, self).__init__()    
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32))) 
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32))) 
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        
        self.feat_dim = self.n_feat_th.shape[1]
        
        self.model_dim = self.feat_dim
        
        self.batch_size = 10
        
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn': #################################################
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time': #####################
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim*4, self.feat_dim*4, self.feat_dim*4, 1) 
        self.probability_score = MergeLayer(self.feat_dim*4, self.feat_dim*4, self.feat_dim*4, 1)
        


        self.lin1 = Linear(self.feat_dim*3, self.feat_dim*3)
        self.lin2 = Linear(self.feat_dim*3, 2)

        self.fix_r = False
        self.init_r = 0.9
        self.decay_interval = 10
        self.decay_r = 0.1
        self.final_r = 0.7 # 0.1

        self.sampling_size = 100

    def forward(self, u_emb, i_emb, i_emb_fake, time, e_emb, k, epoch, training, num_neighbors=10):

        device = self.n_feat_th.device

        # Z_ek
        u_emb_k = u_emb[k][np.newaxis]
        i_emb_k = i_emb[k][np.newaxis]
        time_k = time[k][np.newaxis]
        e_idx_k = (e_emb[k]-1)[np.newaxis]

        t_idx_u_emb_k = self.tem_conv(u_emb_k, time_k, self.num_layers, num_neighbors)
        t_idx_i_emb_k = self.tem_conv(i_emb_k, time_k, self.num_layers, num_neighbors)
        edge_idx_l_th_k = torch.from_numpy(e_idx_k).long().to(device)
        t_idx_e_emb_k = self.edge_raw_embed(edge_idx_l_th_k)

        time_tensor_k = torch.from_numpy(time_k).float().to(device)
        time_tensor_k = torch.unsqueeze(time_tensor_k, dim=1)
        time_encoder_k = self.time_encoder(torch.zeros_like(time_tensor_k)).reshape(1, -1) 

        target_event_emb_k = torch.cat([t_idx_u_emb_k, t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)

        # fake Z_ek
        fake_t_idx_i_emb_k = self.tem_conv(i_emb_fake, time_k, self.num_layers, num_neighbors)
        fake_target_event_emb_k = torch.cat([t_idx_u_emb_k, fake_t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)



        # Z_ej
        one_hop_ngh_node_i_emb, one_hop_ngh_eidx_i_emb, one_hop_ngh_ts_i_emb = self.ngh_finder.get_temporal_neighbor(u_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_i_emb, one_hop_ngh_eidx_i_emb, one_hop_ngh_ts_i_emb = one_hop_ngh_node_i_emb.squeeze(), one_hop_ngh_eidx_i_emb.squeeze(), one_hop_ngh_ts_i_emb.squeeze()
        one_hop_u_emb_i1 = t_idx_u_emb_k.repeat(num_neighbors,1)
        one_hop_i_emb_i1 = self.tem_conv(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, self.num_layers, num_neighbors)
        one_hop_e_idx_i1 = torch.from_numpy(one_hop_ngh_eidx_i_emb).long().to(device)
        one_hop_e_emb_i1 = self.edge_raw_embed(one_hop_e_idx_i1)

        one_hop_time_del_i1 = time_k - one_hop_ngh_ts_i_emb
        one_hop_time_del_i1 = torch.from_numpy(one_hop_time_del_i1).float().to(device)
        one_hop_time_del_i1 = one_hop_time_del_i1.reshape(-1,1)
        one_hop_time_encoder_i1 = self.time_encoder(one_hop_time_del_i1).reshape(num_neighbors,-1) 

        one_hop_event_emb_i1 = torch.cat([one_hop_u_emb_i1, one_hop_i_emb_i1, one_hop_time_encoder_i1, one_hop_e_emb_i1], dim=1)


        two_hop_ngh_node_u_emb, two_hop_ngh_eidx_u_emb, two_hop_ngh_ts_u_emb = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, num_neighbors) 
        two_hop_ngh_node_u_emb, two_hop_ngh_eidx_u_emb, two_hop_ngh_ts_u_emb = two_hop_ngh_node_u_emb.squeeze(), two_hop_ngh_eidx_u_emb.squeeze(), two_hop_ngh_ts_u_emb.squeeze()
        two_hop_u_emb_i1 = self.tem_conv(two_hop_ngh_node_u_emb.reshape(-1), two_hop_ngh_ts_u_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_i_emb_i1 = one_hop_i_emb_i1.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_e_idx_i1 = torch.from_numpy(two_hop_ngh_eidx_u_emb).long().to(device)
        two_hop_e_emb_i1 = self.edge_raw_embed(two_hop_e_idx_i1).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i1 = time_k - two_hop_ngh_ts_u_emb
        two_hop_time_del_i1 = torch.from_numpy(two_hop_time_del_i1).float().to(device)
        two_hop_time_del_i1 = two_hop_time_del_i1.reshape(-1,1)
        two_hop_time_encoder_i1 = self.time_encoder(two_hop_time_del_i1).reshape(num_neighbors*num_neighbors, -1) 
        
        two_hop_event_emb_i1 = torch.cat([two_hop_u_emb_i1, two_hop_i_emb_i1, two_hop_time_encoder_i1, two_hop_e_emb_i1], dim=1)


        one_hop_ngh_node_u_emb, one_hop_ngh_eidx_u_emb, one_hop_ngh_ts_u_emb = self.ngh_finder.get_temporal_neighbor(i_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_u_emb, one_hop_ngh_eidx_u_emb, one_hop_ngh_ts_u_emb = one_hop_ngh_node_u_emb.squeeze(), one_hop_ngh_eidx_u_emb.squeeze(), one_hop_ngh_ts_u_emb.squeeze()
        one_hop_u_emb_i2 = self.tem_conv(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, self.num_layers, num_neighbors)
        one_hop_i_emb_i2 = t_idx_i_emb_k.repeat(num_neighbors,1)
        one_hop_e_idx_i2 = torch.from_numpy(one_hop_ngh_eidx_u_emb).long().to(device)
        one_hop_e_emb_i2 = self.edge_raw_embed(one_hop_e_idx_i2)

        one_hop_time_del_i2 = time_k - one_hop_ngh_ts_u_emb
        one_hop_time_del_i2 = torch.from_numpy(one_hop_time_del_i2).float().to(device)
        one_hop_time_del_i2 = one_hop_time_del_i2.reshape(-1,1)
        one_hop_time_encoder_i2 = self.time_encoder(one_hop_time_del_i2).reshape(num_neighbors,-1) 

        one_hop_event_emb_i2 = torch.cat([one_hop_u_emb_i2, one_hop_i_emb_i2, one_hop_time_encoder_i2, one_hop_e_emb_i2], dim=1)


        two_hop_ngh_node_i_emb, two_hop_ngh_eidx_i_emb, two_hop_ngh_ts_i_emb = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, num_neighbors) 
        two_hop_ngh_node_i_emb, two_hop_ngh_eidx_i_emb, two_hop_ngh_ts_i_emb = two_hop_ngh_node_i_emb.squeeze(), two_hop_ngh_eidx_i_emb.squeeze(), two_hop_ngh_ts_i_emb.squeeze()
        two_hop_u_emb_i2 = one_hop_u_emb_i2.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_i_emb_i2 = self.tem_conv(two_hop_ngh_node_i_emb.reshape(-1), two_hop_ngh_ts_i_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_e_idx_i2 = torch.from_numpy(two_hop_ngh_eidx_i_emb).long().to(device)
        two_hop_e_emb_i2 = self.edge_raw_embed(two_hop_e_idx_i2).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i2 = time_k - two_hop_ngh_ts_i_emb
        two_hop_time_del_i2 = torch.from_numpy(two_hop_time_del_i2).float().to(device)
        two_hop_time_del_i2 = two_hop_time_del_i2.reshape(-1,1)
        two_hop_time_encoder_i2 = self.time_encoder(two_hop_time_del_i2).reshape(num_neighbors*num_neighbors, -1)  
        
        two_hop_event_emb_i2 = torch.cat([two_hop_u_emb_i2, two_hop_i_emb_i2, two_hop_time_encoder_i2, two_hop_e_emb_i2], dim=1)

        target_event_emb_i = torch.cat([one_hop_event_emb_i1, one_hop_event_emb_i2, two_hop_event_emb_i1, two_hop_event_emb_i2], dim=0)

        pos_score_logits = self.affinity_score(target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i) #.squeeze(dim=-1)
        pos_score = self.sampling(pos_score_logits, training) # 여기서 simoid가 들어감..
        pos_score = pos_score.reshape(-1)

        neg_score_logits = self.affinity_score(fake_target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i) #.squeeze(dim=-1)
        neg_score = self.sampling(neg_score_logits, training)
        neg_score = neg_score.reshape(-1)



        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        pos_info_loss = (pos_score * torch.log(pos_score/r + 1e-6) + (1-pos_score) * torch.log((1-pos_score)/(1-r+1e-6) + 1e-6)).mean()
        neg_info_loss = (neg_score * torch.log(neg_score/r + 1e-6) + (1-neg_score) * torch.log((1-neg_score)/(1-r+1e-6) + 1e-6)).mean()
        info_loss = (pos_info_loss + neg_info_loss) / 2

        pos_score = pos_score.reshape(-1,1)
        pos_active_edge_emb = pos_score * target_event_emb_i
        neg_score = neg_score.reshape(-1,1)
        neg_active_edge_emb = neg_score * target_event_emb_i

        pos_active_graph = global_mean_pool(pos_active_edge_emb, batch=None)
        pos_prob = self.probability_score(pos_active_graph, target_event_emb_k).sigmoid()

        neg_active_graph = global_mean_pool(neg_active_edge_emb, batch=None)
        neg_prob = self.probability_score(neg_active_graph, fake_target_event_emb_k).sigmoid()        


        return pos_prob.squeeze(dim=1), neg_prob.squeeze(dim=1), info_loss 

        

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        # pdb.set_trace()
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise) 
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid() 
        else:
            att_bern = (att_log_logit).sigmoid() 
        return att_bern 

    
    def tem_conv(self, u_emb, time, curr_layers, num_neighbors):
        assert(curr_layers >= 0)
        
        device = self.n_feat_th.device
    
        u_emb_th = torch.from_numpy(u_emb).long().to(device) 
        time_th = torch.from_numpy(time).float().to(device) 

        time_th = torch.unsqueeze(time_th, dim=1)

        src_node_t_embed = self.time_encoder(torch.zeros_like(time_th)) 
        src_node_feat = self.node_raw_embed(u_emb_th) 

        if curr_layers == 0:
            return src_node_feat
        else: ######
            src_node_conv_feat = self.tem_conv(u_emb, 
                                           time,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors) 
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    u_emb, 
                                                                    time, 
                                                                    num_neighbors=num_neighbors) 

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = time[:, np.newaxis] - src_ngh_t_batch 
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() 
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() 
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers = curr_layers - 1, 
                                                   num_neighbors=num_neighbors) 
            src_ngh_feat = src_ngh_node_conv_feat.view(len(u_emb), num_neighbors, -1)
            

            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)


            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]


            local, weight = attn_m(src_node_conv_feat, 
                                src_node_t_embed,
                                src_ngh_feat,
                                src_ngh_t_embed, 
                                src_ngn_edge_feat, 
                                mask)

            return local
        
class TGIB_with_fc(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, hidden,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        super(TGIB_with_fc, self).__init__()    
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32))) 
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32))) 
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        
        self.feat_dim = self.n_feat_th.shape[1]
        
        self.model_dim = self.feat_dim
        
        self.batch_size = 10
        
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn': #################################################
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time': #####################
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim*4, self.feat_dim*4, self.feat_dim*4, 1) 
        #self.affinity_score = MergeLayer_with_fc(self.feat_dim*4, self.feat_dim*4, self.feat_dim*3, self.feat_dim*3, self.feat_dim*4, 1) 
        #self.probability_score = MergeLayer(self.feat_dim*4, self.feat_dim*4, self.feat_dim*4, 1)
        self.probability_score = MergeLayer_with_fc(self.feat_dim*4, self.feat_dim*4, self.feat_dim*3, self.feat_dim*3, self.feat_dim*4, 1)
        
        self.lin1 = Linear(self.feat_dim*3, self.feat_dim*3)
        self.lin2 = Linear(self.feat_dim*3, 2)

        self.fix_r = False
        self.init_r = 0.9
        self.decay_interval = 10
        self.decay_r = 0.1
        self.final_r = 0.7 # 0.1

        self.sampling_size = 100

        # self.bottleneck_fc = nn.Sequential(
        #     nn.Linear(self.feat_dim*4, self.feat_dim*4),  
        #     nn.ReLU(),
        #     nn.Dropout(p=drop_out),
        #     nn.Linear(self.feat_dim*3, self.feat_dim*3), 
        #     nn.ReLU(),
        #     nn.Dropout(p=drop_out),
        #     nn.Linear(self.feat_dim*3, self.feat_dim*4), 
        #     nn.ReLU(),
        # )

    def forward(self, u_emb, i_emb, i_emb_fake, time, e_emb, k, epoch, training, num_neighbors=10):

        device = self.n_feat_th.device

        # Z_ek
        u_emb_k = u_emb[k][np.newaxis]
        i_emb_k = i_emb[k][np.newaxis]
        time_k = time[k][np.newaxis]
        e_idx_k = (e_emb[k]-1)[np.newaxis]

        t_idx_u_emb_k = self.tem_conv(u_emb_k, time_k, self.num_layers, num_neighbors)
        t_idx_i_emb_k = self.tem_conv(i_emb_k, time_k, self.num_layers, num_neighbors)
        edge_idx_l_th_k = torch.from_numpy(e_idx_k).long().to(device)
        t_idx_e_emb_k = self.edge_raw_embed(edge_idx_l_th_k)

        time_tensor_k = torch.from_numpy(time_k).float().to(device)
        time_tensor_k = torch.unsqueeze(time_tensor_k, dim=1)
        time_encoder_k = self.time_encoder(torch.zeros_like(time_tensor_k)).reshape(1, -1) 

        target_event_emb_k = torch.cat([t_idx_u_emb_k, t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)

        # fake Z_ek
        fake_t_idx_i_emb_k = self.tem_conv(i_emb_fake, time_k, self.num_layers, num_neighbors)
        fake_target_event_emb_k = torch.cat([t_idx_u_emb_k, fake_t_idx_i_emb_k, time_encoder_k, t_idx_e_emb_k], dim=1)



        # Z_ej
        one_hop_ngh_node_i_emb, one_hop_ngh_eidx_i_emb, one_hop_ngh_ts_i_emb = self.ngh_finder.get_temporal_neighbor(u_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_i_emb, one_hop_ngh_eidx_i_emb, one_hop_ngh_ts_i_emb = one_hop_ngh_node_i_emb.squeeze(), one_hop_ngh_eidx_i_emb.squeeze(), one_hop_ngh_ts_i_emb.squeeze()
        one_hop_u_emb_i1 = t_idx_u_emb_k.repeat(num_neighbors,1)
        one_hop_i_emb_i1 = self.tem_conv(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, self.num_layers, num_neighbors)
        one_hop_e_idx_i1 = torch.from_numpy(one_hop_ngh_eidx_i_emb).long().to(device)
        one_hop_e_emb_i1 = self.edge_raw_embed(one_hop_e_idx_i1)

        one_hop_time_del_i1 = time_k - one_hop_ngh_ts_i_emb
        one_hop_time_del_i1 = torch.from_numpy(one_hop_time_del_i1).float().to(device)
        one_hop_time_del_i1 = one_hop_time_del_i1.reshape(-1,1)
        one_hop_time_encoder_i1 = self.time_encoder(one_hop_time_del_i1).reshape(num_neighbors,-1) 

        one_hop_event_emb_i1 = torch.cat([one_hop_u_emb_i1, one_hop_i_emb_i1, one_hop_time_encoder_i1, one_hop_e_emb_i1], dim=1)


        two_hop_ngh_node_u_emb, two_hop_ngh_eidx_u_emb, two_hop_ngh_ts_u_emb = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_i_emb, one_hop_ngh_ts_i_emb, num_neighbors) 
        two_hop_ngh_node_u_emb, two_hop_ngh_eidx_u_emb, two_hop_ngh_ts_u_emb = two_hop_ngh_node_u_emb.squeeze(), two_hop_ngh_eidx_u_emb.squeeze(), two_hop_ngh_ts_u_emb.squeeze()
        two_hop_u_emb_i1 = self.tem_conv(two_hop_ngh_node_u_emb.reshape(-1), two_hop_ngh_ts_u_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_i_emb_i1 = one_hop_i_emb_i1.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_e_idx_i1 = torch.from_numpy(two_hop_ngh_eidx_u_emb).long().to(device)
        two_hop_e_emb_i1 = self.edge_raw_embed(two_hop_e_idx_i1).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i1 = time_k - two_hop_ngh_ts_u_emb
        two_hop_time_del_i1 = torch.from_numpy(two_hop_time_del_i1).float().to(device)
        two_hop_time_del_i1 = two_hop_time_del_i1.reshape(-1,1)
        two_hop_time_encoder_i1 = self.time_encoder(two_hop_time_del_i1).reshape(num_neighbors*num_neighbors, -1) 
        
        two_hop_event_emb_i1 = torch.cat([two_hop_u_emb_i1, two_hop_i_emb_i1, two_hop_time_encoder_i1, two_hop_e_emb_i1], dim=1)


        one_hop_ngh_node_u_emb, one_hop_ngh_eidx_u_emb, one_hop_ngh_ts_u_emb = self.ngh_finder.get_temporal_neighbor(i_emb_k, time_k, num_neighbors) 
        one_hop_ngh_node_u_emb, one_hop_ngh_eidx_u_emb, one_hop_ngh_ts_u_emb = one_hop_ngh_node_u_emb.squeeze(), one_hop_ngh_eidx_u_emb.squeeze(), one_hop_ngh_ts_u_emb.squeeze()
        one_hop_u_emb_i2 = self.tem_conv(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, self.num_layers, num_neighbors)
        one_hop_i_emb_i2 = t_idx_i_emb_k.repeat(num_neighbors,1)
        one_hop_e_idx_i2 = torch.from_numpy(one_hop_ngh_eidx_u_emb).long().to(device)
        one_hop_e_emb_i2 = self.edge_raw_embed(one_hop_e_idx_i2)

        one_hop_time_del_i2 = time_k - one_hop_ngh_ts_u_emb
        one_hop_time_del_i2 = torch.from_numpy(one_hop_time_del_i2).float().to(device)
        one_hop_time_del_i2 = one_hop_time_del_i2.reshape(-1,1)
        one_hop_time_encoder_i2 = self.time_encoder(one_hop_time_del_i2).reshape(num_neighbors,-1) 

        one_hop_event_emb_i2 = torch.cat([one_hop_u_emb_i2, one_hop_i_emb_i2, one_hop_time_encoder_i2, one_hop_e_emb_i2], dim=1)


        two_hop_ngh_node_i_emb, two_hop_ngh_eidx_i_emb, two_hop_ngh_ts_i_emb = self.ngh_finder.get_temporal_neighbor(one_hop_ngh_node_u_emb, one_hop_ngh_ts_u_emb, num_neighbors) 
        two_hop_ngh_node_i_emb, two_hop_ngh_eidx_i_emb, two_hop_ngh_ts_i_emb = two_hop_ngh_node_i_emb.squeeze(), two_hop_ngh_eidx_i_emb.squeeze(), two_hop_ngh_ts_i_emb.squeeze()
        two_hop_u_emb_i2 = one_hop_u_emb_i2.repeat(1,num_neighbors).reshape(num_neighbors*num_neighbors, -1)
        two_hop_i_emb_i2 = self.tem_conv(two_hop_ngh_node_i_emb.reshape(-1), two_hop_ngh_ts_i_emb.reshape(-1), self.num_layers, num_neighbors)
        two_hop_e_idx_i2 = torch.from_numpy(two_hop_ngh_eidx_i_emb).long().to(device)
        two_hop_e_emb_i2 = self.edge_raw_embed(two_hop_e_idx_i2).reshape(num_neighbors*num_neighbors, -1)

        two_hop_time_del_i2 = time_k - two_hop_ngh_ts_i_emb
        two_hop_time_del_i2 = torch.from_numpy(two_hop_time_del_i2).float().to(device)
        two_hop_time_del_i2 = two_hop_time_del_i2.reshape(-1,1)
        two_hop_time_encoder_i2 = self.time_encoder(two_hop_time_del_i2).reshape(num_neighbors*num_neighbors, -1)  
        
        two_hop_event_emb_i2 = torch.cat([two_hop_u_emb_i2, two_hop_i_emb_i2, two_hop_time_encoder_i2, two_hop_e_emb_i2], dim=1)

        target_event_emb_i = torch.cat([one_hop_event_emb_i1, one_hop_event_emb_i2, two_hop_event_emb_i1, two_hop_event_emb_i2], dim=0)

        pos_score_logits = self.affinity_score(target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i) #.squeeze(dim=-1)
        pos_score = self.sampling(pos_score_logits, training) # 여기서 simoid가 들어감..
        pos_score = pos_score.reshape(-1)

        neg_score_logits = self.affinity_score(fake_target_event_emb_k.repeat(2*num_neighbors*(num_neighbors+1),1), target_event_emb_i) #.squeeze(dim=-1)
        neg_score = self.sampling(neg_score_logits, training)
        neg_score = neg_score.reshape(-1)



        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        pos_info_loss = (pos_score * torch.log(pos_score/r + 1e-6) + (1-pos_score) * torch.log((1-pos_score)/(1-r+1e-6) + 1e-6)).mean()
        neg_info_loss = (neg_score * torch.log(neg_score/r + 1e-6) + (1-neg_score) * torch.log((1-neg_score)/(1-r+1e-6) + 1e-6)).mean()
        info_loss = (pos_info_loss + neg_info_loss) / 2

        pos_score = pos_score.reshape(-1,1)
        pos_active_edge_emb = pos_score * target_event_emb_i
        neg_score = neg_score.reshape(-1,1)
        neg_active_edge_emb = neg_score * target_event_emb_i

        pos_active_graph = global_mean_pool(pos_active_edge_emb, batch=None)
        pos_prob = self.probability_score(pos_active_graph, target_event_emb_k).sigmoid()

        neg_active_graph = global_mean_pool(neg_active_edge_emb, batch=None)
        neg_prob = self.probability_score(neg_active_graph, fake_target_event_emb_k).sigmoid()        


        return pos_prob.squeeze(dim=1), neg_prob.squeeze(dim=1), info_loss 

        

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        # pdb.set_trace()
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise) 
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid() 
        else:
            att_bern = (att_log_logit).sigmoid() 
        return att_bern 

    
    def tem_conv(self, u_emb, time, curr_layers, num_neighbors):
        assert(curr_layers >= 0)
        
        device = self.n_feat_th.device
    
        u_emb_th = torch.from_numpy(u_emb).long().to(device) 
        time_th = torch.from_numpy(time).float().to(device) 

        time_th = torch.unsqueeze(time_th, dim=1)

        src_node_t_embed = self.time_encoder(torch.zeros_like(time_th)) 
        src_node_feat = self.node_raw_embed(u_emb_th) 

        if curr_layers == 0:
            return src_node_feat
        else: ######
            src_node_conv_feat = self.tem_conv(u_emb, 
                                           time,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors) 
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    u_emb, 
                                                                    time, 
                                                                    num_neighbors=num_neighbors) 

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = time[:, np.newaxis] - src_ngh_t_batch 
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() 
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() 
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers = curr_layers - 1, 
                                                   num_neighbors=num_neighbors) 
            src_ngh_feat = src_ngh_node_conv_feat.view(len(u_emb), num_neighbors, -1)
            

            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)


            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]


            local, weight = attn_m(src_node_conv_feat, 
                                src_node_t_embed,
                                src_ngh_feat,
                                src_ngh_t_embed, 
                                src_ngn_edge_feat, 
                                mask)

            return local


