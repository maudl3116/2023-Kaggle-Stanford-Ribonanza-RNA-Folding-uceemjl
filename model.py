# Adapted from kaggler iafoss notebook (https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb)

import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor


class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        
        self.emb = nn.Embedding(4,dim)
        self.nhead = dim//head_size
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)   
        
        self.proj_out = nn.Linear(dim,2)
        
        # changed the positional encoding (from sinusoidal to the one used in AlphaFold)
        self.linear_pair = nn.Linear(1, self.nhead)
        self.relpos_k = 32 
        self.num_bins = 2 * self.relpos_k + 1
        self.linear_relpos = nn.Linear(self.num_bins, self.nhead)
        
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max() 
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]    
        bppm = x0['bppm'][:,:Lmax,:Lmax]  
        
        x = self.emb(x)
        
        pair_emb = self.linear_pair(bppm[...,None])   # (batch, L, L, nhead)
        
        # positional encoding 
        max_rel_res = self.relpos_k
        res_id = torch.arange(Lmax, device=x.device).unsqueeze(0)  # (1, L)
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res   # (1, L, L)
        
        pos_enc = self.linear_relpos(
                nn.functional.one_hot(rp, num_classes=self.num_bins).float())  # (1, L, L, num_bins)

        pair_emb += pos_enc     # (batch, L, L, nhead)
        
        z = self.transformer(x, mask = pair_emb.permute(0,3,1,2).reshape(-1,Lmax, Lmax), src_key_padding_mask=~mask)  

        y = self.proj_out(z)
        
        return y