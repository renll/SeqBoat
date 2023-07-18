# Copyright (c) Liliang Ren.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import math
from inspect import isfunction
from math import pi, log

from torch import nn, einsum
from einops import rearrange, repeat, pack, unpack
from typing import Optional, Tuple, Union


# constant
TOKEN_SELF_ATTN_VALUE = -5e4


def eval_index(x, dim):
    strx = "x["
    if dim<0:
        xlen = len(x.shape) +dim
    else:
        xlen = dim
    for _ in range(xlen):
        strx +=":,"
    strx += "1:]"    
    return eval(strx)

def compress_seq(q,index_q, max_sl, dim = -2):
    shape = q.shape[:dim] + (max_sl+1,) + q.shape[dim+1:] 
    ori_dtype = q.dtype
    if ori_dtype == torch.half:
        mem_dtype = torch.float
        q = q.float()
    else:
        mem_dtype = q.dtype
         
    new_q = torch.zeros(shape, device = q.device, dtype = mem_dtype)
    new_q.scatter_(dim,index_q, q)
    if max_sl>0:
        new_q = eval_index(new_q,dim)
    else:
        new_q = 0*new_q
    return new_q.type(ori_dtype)

def extract(h, index_q):
    h = F.pad(h, (0,0,1,0))
    h = torch.gather(h.float(),-2,index_q.expand(-1,-1,-1,h.shape[-1])).type_as(h)
    return h


def calc_avg_attn_span(a,win,causal=False):
    #a: ..., q
    #win: int
    seq_len = a.shape[-1]
    pad_mask = (a!=0)
    wind_mask = torch.ones([seq_len,seq_len]).to(a).bool()
    wind_mask = ~(torch.triu(wind_mask,diagonal=win+1) +torch.tril(wind_mask,diagonal=-win-1))
    q=a.unsqueeze(-1)
    k= a.unsqueeze(-2)
    dist = (q-k).abs()
    dist *= pad_mask.unsqueeze(-1)
    dist *= pad_mask.unsqueeze(-2)
    if causal:
        dist = torch.tril(dist)
    dist *=wind_mask
    avg_attn_span = dist.sum(-1)/((dist!=0).sum(-1)+1e-8)
    return avg_attn_span


def look_around_2d(x,window_size=4,backward=1,forward=1):
    #x: b, k ,c, c
    b,k,c,c = x.shape
    n = x.shape[-1]//window_size
    x = x.view( b, k, n, window_size, n,window_size).transpose(-3,-2)
    padded_x = F.pad(x, (0,0,0,0, backward, forward), value = 0)
    def get_diag(offset):
      return torch.diagonal(padded_x, offset=offset, dim1=-4,dim2=-3).transpose(-1,-2).transpose(-3,-2)
    x_list = [get_diag(z) for z in range(backward+forward+1) ]
    return torch.cat(x_list,-1)


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1 )/ dim ** 0.5)
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale
    


def get_gumbel_weights(logits, tau):
    EPS = torch.finfo(torch.float32).tiny
    uniforms = torch.empty_like(logits).float().uniform_().clamp_(EPS, 1 - EPS)
    gumbels = uniforms.log().neg().log().neg()
    gumbels = gumbels.type(logits.dtype).to(logits.device)
    weights = (gumbels + logits)/tau
    return weights

def gumbel_sampling(logits, tau, dim=-1, hard=False, sigmoid=False):
       
    weights = get_gumbel_weights(logits, tau)
    if sigmoid:
        y_soft = torch.sigmoid(weights)
    else:    
        y_soft = F.softmax(weights , dim = dim )

    if hard:
        if sigmoid:
            y_hard = weights > 0
        else:
            index = weights.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(weights, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        y_soft = (y_hard- y_soft).detach()+ y_soft
    return y_soft





def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)

class LatentConfigurator(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        hard = False,
        no_proj = True,
        max_steps = 180000,
        init_temp_scale = 0.3,
    ):

        super().__init__()

        self.input_dim = dim
        self.num_vars = num_vars
        self.hard = hard
        self.no_proj = no_proj

        proj_dim =  num_vars 
        self.proj_dim = proj_dim

        if not self.no_proj:
            self.weight_proj = nn.Linear(self.input_dim, proj_dim, bias = True)
            Normal(self.weight_proj.weight, mean=0, std=0.02)
            nn.init.zeros_(self.weight_proj.bias)
        
        self.use_sigmoid = proj_dim ==1
        self.learned_temp = True
        if self.learned_temp:
            self.temp = nn.Parameter(torch.ones(1)*math.log(init_temp_scale * self.input_dim**0.5))
        else:
            self.max_temp, self.min_temp = [2,0.5]
            self.curr_temp = self.min_temp
            prop = 0.5
            self.temp_decay = (self.min_temp / self.max_temp)**(1/max_steps/prop) 

    def set_num_updates(self, num_updates):
        
        if self.learned_temp:
            return
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def forward(self, x, deterministic = True, act_bias = 0):

        result = {}


        if self.learned_temp:
            temp = self.temp.exp()   
        else:
            temp = self.curr_temp

        if not self.no_proj:
            x = self.weight_proj(x)
        

        if act_bias !=0:
            x = x + torch.tensor([act_bias,0]).to(x)
            

        if self.learned_temp:
            if self.use_sigmoid:
                x = torch.sigmoid(x/temp,dtype=torch.float32).type_as(x)
            else:
                x = F.softmax(x/temp, dim=-1,dtype=torch.float32).type_as(x)
        else:
            if self.training:
                if self.use_sigmoid:
                    x = gumbel_sampling(x.float(), tau=temp, hard=self.hard, sigmoid = True).type_as(x)
                else:
                    x = F.gumbel_softmax(x.float(), tau=temp, hard=self.hard).type_as(x)
            else:
                if deterministic:
                    if self.use_sigmoid:
                        hard_x = x>0
                    else:
                        _, k = x.max(-1,keepdim=True)
                        hard_x = x.new_zeros(*x.shape).scatter_(-1, k, 1.0)
                    
                    x = hard_x.type_as(x)

                else:
                    if self.use_sigmoid:
                        x = gumbel_sampling(x.float(), tau=temp, hard=self.hard, sigmoid = True).type_as(x)
                    else:
                        x = F.gumbel_softmax(x.float(), tau=temp, hard=self.hard).type_as(x)
                    
        result["temp"] = temp
        result["probs"] = x 
 
        return result



def Normal(tensor, mean=0.0, std=0.02):
    #return nn.init.normal_(tensor, mean=mean, std=std)
    return truncated_normal_(tensor, mean, std)

def truncated_normal_(tensor, mean=0.0, std=0.02):
   
    tensor=torch.nn.init.trunc_normal_(tensor, mean, std, -2*std, 2*std)
   
    return tensor


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''unfold T x B x C to T x B x C x K'''
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
    else:
        x = x.unsqueeze(3)
    return x


class FastRelativePositionalBias(nn.Module):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        Normal(self.rel_pos_bias, mean=0.0, std=std)

    def forward(self, seq_len):
        # seq_len * 2 -1
        if seq_len<= self.max_positions:
            start_pos = self.max_positions - seq_len
            b = self.rel_pos_bias[start_pos: start_pos + 2*seq_len - 1]
        else:
            delta = seq_len - self.max_positions
            b = self.rel_pos_bias
            prefix = self.rel_pos_bias[:1].expand(delta)
            postfix = self.rel_pos_bias[-1:].expand(delta)
            b = torch.cat([prefix,b,postfix], dim =-1)

        # seq_len * 3 - 1
        t = F.pad(b, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        
        return t
        #non deterministic!
        # x = F.pad(b.contiguous(), ( 0,seq_len - 1 ), value=0)
        # t = x.as_strided((seq_len,seq_len), (1, 1),0).flip(0)
        # return t
        
        # b = b[:,None,None]
        # t1 = unfold1d(b,b.shape[0],0)
        # t1 = t1[:seq_len,0,0,:seq_len]
        # return 

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)



# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb_old(freqs, t, start_index = 0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb_old(rotations, t, start_index = start_index)



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    #freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq)



# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        #self.relative_attention_bias = nn.Embedding(num_buckets, 1)
        self.relative_attention_bias = nn.Parameter(torch.Tensor(num_buckets,1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        Normal(self.relative_attention_bias, mean=0.0, std=std)
        
    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = F.relu(n,inplace=True)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_pos,k_pos,causal=False):
        rel_pos = k_pos.unsqueeze(-2) - q_pos.unsqueeze(-1)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = causal, num_buckets = self.num_buckets, max_distance = self.max_distance) 
        values = torch.matmul(F.one_hot(rp_bucket,num_classes=self.num_buckets).to(self.relative_attention_bias).detach(),
                                self.relative_attention_bias)
        #values = self.relative_attention_bias(rp_bucket)
        bias = values.squeeze(-1)
        return bias * self.scale