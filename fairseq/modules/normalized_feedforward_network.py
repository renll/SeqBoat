# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout, FairseqFeatureDropout
from fairseq.modules.sequence_norm import SequenceNorm
#from .pam_utils import compress_tick, compress_seq, sparse_self_att, PAM_loss, GumbelVectorQuantizer, LocalAttention, default, RotaryEmbedding, GumbelCtxConfigurator, calc_expcum_attn

from .seqboat_utils import Normal

class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        export=False,
        trunc_norm = False,
    ):
        super().__init__()

        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = activation
        self.activation = utils.get_activation_fn(activation)

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)

        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        
        
        self.q_route = False
        if self.q_route:
            self.num_options = 2
            self.density_threshold = 0.9
            self.opt_dim = self.num_options -1
            self.q_router = GumbelVectorQuantizer(
                dim = embed_dim,
                num_vars = self.num_options,
                temp = (5, 0.5, 1-3e-5),
                groups = 1,
                combine_groups = True,#False,
                vq_dim = embed_dim,
                hard = False,
                only_probs = True,
                kp = 50,
                no_proj= True
            ) 
            self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim+self.opt_dim)
        else:
            self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, embed_dim)
        if trunc_norm:
            self.trunc_reset_parameters()
        else:
            self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc2.bias, 0.0)

    def trunc_reset_parameters(self):
        std = 0.02
        Normal(self.fc1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc1.bias, 0.0)

        Normal(self.fc2.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, x):
        residual = x

        if self.prenorm:
            x = self.norm(x)
        if self.q_route:
            base = self.fc1(x)
            x, u = torch.split(base, [self.hidden_dim, self.opt_dim],dim=-1)
            result = self.q_router(u, mask_p = None, deterministic=True)
            px = result["gumbel_probs"]
            q_p, q_ind = px.max(dim=-1, keepdim = True) #b*h,q,1
            mask_q = (q_ind==1)#b*h,q,1
            q_probs = (q_p * mask_q)
            
        else:
            x = self.fc1(x)
        x = self.activation(x)
        x = self.hidden_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.q_route:
            x = q_probs*x
        x = x + residual

        if not self.prenorm:
            x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return 'edim={}, hdim={}, act={}, prenorm={}'.format(self.embedding_dim, self.hidden_dim, self.act_fn, self.prenorm)
