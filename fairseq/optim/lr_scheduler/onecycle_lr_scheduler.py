# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from . import FairseqLRScheduler, register_lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR

@register_lr_scheduler('one_cycle')
class OneCycleSchedule(FairseqLRScheduler):
    """Assign LR based on a triangular cyclical schedule.

    See https://arxiv.org/pdf/1506.01186.pdf for details.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with triangular.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        lr = args.lr[0]
        mult = 1
        self.sch = OneCycleLR(optimizer._optimizer, lr, total_steps= int(mult*args.max_update+2), epochs=None, 
                              steps_per_epoch=None, pct_start=args.pct_start, 
                              anneal_strategy=args.anneal_strategy, cycle_momentum=True, 
                              base_momentum=0.85, max_momentum=0.95, 
                              div_factor=25.0, 
                              final_div_factor=10000.0, 
                              three_phase=args.three_phase, 
                              last_epoch=- 1, 
                              verbose=False)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--anneal-strategy', default='cos', type=str)
        parser.add_argument('--pct-start', default=0.3, type=float)
        parser.add_argument('--three-phase', action='store_true')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        self.sch.last_epoch=num_updates
        self.sch.step()
        return self.optimizer.get_lr()
