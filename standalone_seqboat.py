# Copyright (c) Liliang Ren.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch import einsum
from einops import rearrange, repeat, pack, unpack
import math

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
    new_q = torch.zeros(shape, device = q.device, dtype = q.dtype)
    new_q.scatter_(dim,index_q, q)
    if max_sl>0:
        new_q = eval_index(new_q,dim)
    else:
        new_q = 0*new_q
    return new_q.type(q.dtype)

def extract(h, index_q):
    h = F.pad(h, (0,0,1,0))
    h = torch.gather(h,-2,index_q.expand(-1,-1,h.shape[-1]))
    return h


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


def Normal(tensor, mean=0.0, std=0.02):
    return torch.nn.init.trunc_normal_(tensor, mean, std, -2*std, 2*std)


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
    
    
class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None
        self.reset_parameters_trunc()
      
    def reset_parameters_trunc(self):
        with torch.no_grad():
            # delta & alpha
            Normal(self.delta, mean=0.0, std=0.2)
            Normal(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            Normal(self.beta, mean=0.0, std=0.02)
            self.beta.add_(val)
            # gamma & omega
            Normal(self.gamma, mean=0.0, std=1.0)
            Normal(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
        self,
        x,
        padding_mask = None,
        incremental_state = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = out + residual
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                x = F.pad(x, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = out.permute(2, 0, 1) + residual

        return out


class LatentConfigurator(nn.Module):
    def __init__(
        self,
        dim,
        proj_dim,
        hard = False,
        no_proj = True,
        init_temp_scale = 1.0,
    ):

        super().__init__()
        self.input_dim = dim
        self.num_vars = proj_dim
        self.hard = hard
        self.no_proj = no_proj
        self.proj_dim = proj_dim

        if not self.no_proj:
            self.weight_proj = nn.Linear(self.input_dim, proj_dim, bias = True)
            Normal(self.weight_proj.weight, mean = 0, std = 0.02)
            nn.init.zeros_(self.weight_proj.bias)
        
        self.use_sigmoid = proj_dim ==1
        self.temp = nn.Parameter(torch.ones(1) * math.log(init_temp_scale * self.input_dim ** 0.5))

    def forward(self, x):

        temp = self.temp.exp()   

        if not self.no_proj:
            x = self.weight_proj(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x / temp,dtype=torch.float32).type_as(x)
        else:
            x = F.softmax(x / temp, dim=-1,dtype=torch.float32).type_as(x)   
            
        return x


class SequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True):
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def forward(self, x):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.norm(x)
    
    
# helper functions

def exists(val):
    return val is not None

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


# rotary embedding helper functions

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq)

    

class SeqBoatLayer(nn.Module):
    """A standalone version of SeqBoat.
    
    Currently does not support KV caching for efficient O(1) per-step decoding.
    """
    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_activation='relu2',
        bidirectional=False,
        window_size=256, # single-sided window size
        truncation=None,
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        rel_pos_bias='simple',
        max_positions=1024,
        init_temp_scale=1.0,
        local_pos = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = F.silu
        self.attention_activation = attention_activation
        self.truncation = truncation
        self.causal = not bidirectional
        self.dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.local_pos = local_pos and rel_pos_bias == 'simple'
        self.density_threshold = 1
        self.window_size = window_size

        self.scaling = self.zdim ** -0.5 
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine,)
            
        self.ssm = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.rel_pos_type = rel_pos_bias
        self.max_positions = max_positions
        
        if rel_pos_bias == 'simple':
            max_positions = max_positions
            if self.local_pos:
                max_positions = self.window_size*3 # Most 3x with non-causal
            self.rel_pos_bias = FastRelativePositionalBias(max_positions)

        elif rel_pos_bias == 'rotary':
            self.max_pos = max_positions
            self.freqs_cis = precompute_freqs_cis(self.zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        num_options = 2
        self.opt_dim = num_options 
        self.q_router = LatentConfigurator(dim = embed_dim, proj_dim= num_options, init_temp_scale= init_temp_scale) 
       
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + hdim)
        self.hx_proj = nn.Linear(embed_dim, self.opt_dim + embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.reset_parameters()


    def reset_parameters(self):
        std = 0.02
        Normal(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)
        Normal(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)
        Normal(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

        Normal(self.hx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.hx_proj.bias, 0.0)



    def local_attention(self, q, k, v, attn_bias, mask, causal):
        
        # attn_bias: b,k,c,c
        autopad, pad_value, window_size = True, -1, self.window_size
        exact_windowsize = True
        if causal:
            look_backward = 1
            look_forward = 0
        else:
            look_backward = 1
            look_forward = 1
            
        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding
        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))
            
        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size


        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)


        # calculate positions for masking
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
        
        if self.local_pos:
            sim = sim + self.rel_pos_bias(sim.shape[-1])[:sim.shape[-2],:]
        elif exists(attn_bias):
            if autopad:
                _,attn_bias = pad_to_multiple(attn_bias, self.window_size, dim = -2)
                _,attn_bias = pad_to_multiple(attn_bias, self.window_size, dim = -1)
            attn_bias = look_around_2d(attn_bias,window_size,look_backward,look_forward)
            ab,ak,aw,ac,av = attn_bias.shape
            sim = sim + attn_bias.view(-1,aw,ac,av)

        if self.attention_activation =="softmax":
            mask_value =  TOKEN_SELF_ATTN_VALUE
        else:
            mask_value = 0

        if causal:
            causal_mask = bq_t < bq_k

            if exact_windowsize:
                max_causal_window_size = (self.window_size * look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value
        if not causal and exact_windowsize:
            max_backward_window_size = (self.window_size * look_backward)
            max_forward_window_size = (self.window_size * look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in
        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            sim = sim.masked_fill(~mask, mask_value)
            del mask
        
        # attention
        if self.attention_activation == 'softmax':
            attn = F.softmax(sim, dim=-1,dtype=torch.float32).type_as(sim)
        elif self.attention_activation == 'relu2':
            attn = torch.square(F.relu(sim)).type_as(sim)  
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        attn = self.attention_dropout(attn)

        # aggregation
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        
        return out



    def element_attention(self, q, k, v, padding_mask, tick,compress, seq_len):
        # padding_mask: b,c
        
        slen = k.size(-2)
        causal = self.causal

        if self.local_pos:
            bias = None
        elif self.rel_pos_type == 'rotary':
            q = apply_rotary_emb(q, tick)
            k = apply_rotary_emb(k, tick)
            bias = None
        else:
            # C x C
            bias = self.rel_pos_bias(seq_len)

            if slen != q.size(-2):
                assert q.size(-2) == 1
                # 1 x C
                bias = bias[-1:]
                       
            if compress:
                bias = F.pad(bias,(1,0,1,0))
                bias = bias[tick]
                bias = torch.gather(bias, -1, tick[:,None,:].expand(-1, bias.shape[-2],-1)).unsqueeze(1)
            else:
                bias = bias[None,None,:,:]
        
        # bias : # b k c c
        if self.attention_activation == 'softmax':
            q = q * self.scaling
            if padding_mask is not None:
                padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
                padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
        else:
            if causal:
                lengths = self.window_size
            else:
                lengths = self.window_size*2
            # q: b c h
            # tick: b c 
            q =q / lengths

        h = self.local_attention(q, k, v, bias, ~padding_mask, causal) 

        return h


    def forward(
        self,
        x,
        padding_mask = None,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()

        residual = x
        if self.prenorm:
           x = self.norm(x)
           
        # L x B x D
        mx = self.ssm(x, padding_mask)
        mx = self.activation(mx)
        mx = self.dropout(mx)
        mx = mx.transpose(0,1)
        
        tick=None
        if self.rel_pos_type == 'rotary': 
            if seq_len > self.max_pos:
                self.max_pos = seq_len  
                self.freqs_cis=precompute_freqs_cis(self.zdim, seq_len).to(x.device)
            else:
                self.freqs_cis = self.freqs_cis.to(x.device)
            tick = self.freqs_cis[None, :seq_len, :].expand(bsz,-1,-1)

        
        hu = self.hx_proj(mx)    
        u, hx= torch.split(hu, [self.opt_dim, self.embed_dim], dim=-1)

        if self.rel_pos_type == 'simple' and not self.local_pos:
            tick = torch.arange(1,1+seq_len,device =mx.device)[None,:,None].expand(bsz,-1,-1)

        px = self.q_router(u)
        
        q_p, q_ind = px.max(dim=-1, keepdim = True) # B x C x 1
        mask_q = (q_ind==1) # B x C x 1
                
        if padding_mask is not None:
            pad_mask = ~padding_mask.unsqueeze(-1).bool()
            mask_q *= pad_mask

        q_probs = (q_p * mask_q)
        max_sl = mask_q.sum(-2).max()
        density = max_sl/mask_q.shape[-2]
        compress = density < self.density_threshold # Set compress to be always true if you meet NCCL timeout problem during distributed training
        
        index_q = None
        if compress:
            
            index_q = mask_q * torch.cumsum(mask_q.long(),dim=-2)
            mx = compress_seq(mx,index_q.expand(-1,-1,mx.shape[-1]), max_sl ,dim = -2) # B x C x D  
            if not self.local_pos:
                tick = compress_seq(tick,index_q.expand(-1,-1,tick.shape[-1]), max_sl ,dim = -2) # B x C x D

            pad_mask = compress_seq(mask_q,index_q.expand(-1,-1,1), max_sl ,dim = -2) # B x C x 1  
            padding_mask = ~pad_mask.squeeze(-1)
        else:
            
            padding_mask = ~mask_q.squeeze(-1)
        
        if self.rel_pos_type == 'simple' and not self.local_pos:
            tick = tick.squeeze(-1)   
        # B x L x E
        base = self.mx_proj(mx)
        z, r, v = torch.split(self.activation(base), [self.zdim, self.hdim, self.hdim], dim=-1)
        # B x L x S -> B x L x 1 x S -> B x L x 2 x S
        z = z.unsqueeze(-2) * self.gamma + self.beta
        # B x L x 2 x S -> B x L x S
        q, k = torch.unbind(z, dim=-2)
        v = self.hidden_dropout(v)
        h = self.element_attention(q, k, v, padding_mask, tick, compress, seq_len)

        h = self.h_proj(h*r)
        if not self.causal:
            h = self.dropout(h)

        if compress:
            h = extract(h, index_q)
        h = h*q_probs
        
        h = h.view(bsz, -1, self.embed_dim).transpose(0, 1)
        hx = hx.view(bsz, -1, self.embed_dim).transpose(0, 1)
            
        if not self.causal:
            out = hx + h + residual
        else:
            out = self.dropout(hx + h) + residual
            
        out = self.activation(out)
        if not self.prenorm:
            out = self.norm(out)    
             
        return out

class SeqBoatModel(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=512,
        n_layers=6,
        dropout=0.1,
        prenorm=False,
        max_positions = 1024,
        d_qk = 96,
        d_ema = 16,
        mem_size = 128,
        bidirectional=True,  
        init_temp_scale=1.0,
        norm_type = 'batchnorm',  
        rel_pos_bias='simple', 
        attention_activation='relu2',
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)
        
        self.seq_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.seq_layers.append(
                SeqBoatLayer(d_model,d_qk,d_model*2,d_ema,
                                bidirectional=bidirectional,window_size=mem_size, 
                                prenorm=prenorm,dropout=dropout,norm_type=norm_type,
                                init_temp_scale=init_temp_scale, max_positions=max_positions,
                                rel_pos_bias=rel_pos_bias, 
                                attention_activation=attention_activation,
                            )
            )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        if prenorm:
            self.final_norm = SequenceNorm(norm_type, d_model)
        else:
            self.final_norm = None
        self.reset_parameters()


    def reset_parameters(self):
        std = 0.02
        Normal(self.encoder.weight, mean=0.0, std=std)
        nn.init.constant_(self.encoder.bias, 0.0)
        Normal(self.decoder.weight, mean=0.0, std=std)
        nn.init.constant_(self.decoder.bias, 0.0)
        
    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(0, 1)  # (B, L, d_model) -> (L, B, d_model)
        for layer in self.seq_layers:
            x = layer(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        x = x.transpose(0, 1)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


if __name__=="__main__":
    torch.manual_seed(1)
    
    layer1 = SeqBoatLayer(128,128,256,16,bidirectional=True,window_size=256, norm_type='batchnorm', max_positions=4096, truncation=4096)
    m1 = SeqBoatModel(128,max_positions=4096)
    x = torch.rand(4096,22,128)
    
    print(layer1(x))
    print(m1(x))
    
