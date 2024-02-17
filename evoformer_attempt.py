import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor

import copy
from typing import Optional, Any, Union, Callable

import warnings
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from inspect import isfunction



class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.nhead = dim//head_size
        self.transformer = MyEvoFormer(
            MyTransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)  
        
        self.proj_out = nn.Linear(dim,2)
        
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

        pair_emb += pos_enc
        
        z = self.transformer(x, mask = pair_emb, src_key_padding_mask=~mask)   
        
        return self.proj_out(z)

class MyEvoFormer(Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        
        for i in range(num_layers):
            self.layers[i].number = i 
        
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = False
        self.mask_check = mask_check


    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """

        for mod in self.layers:
            src, mask = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            src = self.norm(src)

        return src
    
class MyTransformerEncoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.nhead = nhead
        self.d_model = d_model
        self.outer_mean = OuterMean(d_model,nhead)
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.norm_pair = LayerNorm(nhead, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.linear1_pair = Linear(nhead, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout_pair = Dropout(dropout)
        self.linear2_pair = Linear(dim_feedforward, nhead, bias=bias, **factory_kwargs)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        
        L = src_mask.shape[1]

        x = x + self._sa_block(self.norm1(x), src_mask.permute(0,3,1,2).reshape(-1, L, L), src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))

        # update the mask (bppm)
        if self.number%2 == 0:

            src_mask = src_mask + self.outer_mean(x)    
            src_mask = src_mask + self._ff_block_pair(self.norm_pair(src_mask))
        
        return x, src_mask


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=False)[0]

        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # same as AF2 except they have GeGLU instead.
        return self.dropout2(x)
    
    def _ff_block_pair(self, x: Tensor) -> Tensor:
        x = self.linear2_pair(self.dropout(self.activation(self.linear1_pair(x))))  # same as AF2 except they have GeGLU instead.
        return self.dropout_pair(x)

   #==============================================================================================================================
# New functionalities for communication
#==============================================================================================================================

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
#         hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
#         self.right_proj = nn.Linear(dim, hidden_dim)
        
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.layer_norm_out = LayerNorm(hidden_dim)

    def forward(self, x, mask = None):
        x = self.norm(x)
        left = self.left_proj(x)
#         right = self.right_proj(x)
        outer =  left[:, :, None, :] * left[:, None, :, :]
        #rearrange(left, 'b i d -> b i () d') * rearrange(right, 'b j d -> b () j d')

        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
#             mask = rearrange(mask, 'b i -> b i () ()') * rearrange(mask, 'b j -> b () j ()')
            mask = mask[:,:,None,None] * mask[:,None,:,None]
            outer = outer.masked_fill(~mask, 0.)

        return self.proj_out(self.layer_norm_out(self.act(outer)))
    
#==============================================================================================================================
# CLUTTER 
#==============================================================================================================================
def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
        
    
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )