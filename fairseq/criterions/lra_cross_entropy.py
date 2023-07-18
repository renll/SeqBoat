# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('lra_cross_entropy')
class LRACrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss_weights = None if loss_weights is None else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(sample)
        losses = []
        loss, correct = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        losses.append(loss)
        
        if self.loss_weights is not None and hasattr(model, "get_extra_losses"):
            extra_losses = model.get_extra_losses(net_output)
            if extra_losses is not None:
                if torch.is_tensor(extra_losses):
                    extra_losses = [extra_losses]
                if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                    self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
                assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
                for p, coef in zip(extra_losses, self.loss_weights):
                    if coef != 0 and p is not None:
                        p = coef * p.float() * sample_size
                        loss += p
                        losses.append(p)
        
        logging_output = {
            'loss': loss.data,
            'ncorrects': correct.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        
        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, targets, reduction='sum')
        preds = torch.argmax(lprobs, 1)
        correct = (preds == targets).sum()
        return loss, correct

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if len(logging_outputs) > 0 and 'ncorrects' in logging_outputs[0]:
            ncorrects = sum(log.get('ncorrects', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrects / nsentences, nsentences, round=3)

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / (sample_size * math.log(2)+1e-10), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
