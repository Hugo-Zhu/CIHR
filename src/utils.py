import torch
from torch import nn
from alegant import logger
from torch.optim.lr_scheduler import _LRScheduler


class ALayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ALayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.linear_gamma = nn.Linear(normalized_shape, 1)
        self.linear_beta = nn.Linear(normalized_shape, 1)

    def forward(self, attribute_embeddings, input):
        batch_size, _, hidden_size = attribute_embeddings.shape
        _, seq_len, hidden_size = input.shape
        
        normalized_output = super(ALayerNorm, self).forward(input)
        gamma = self.linear_gamma(attribute_embeddings).view(batch_size, -1, 1, 1)
        beta = self.linear_beta(attribute_embeddings).view(batch_size, -1, 1, 1)
        normalized_output = normalized_output.view(batch_size, -1, seq_len, hidden_size)
        output = normalized_output * (1+gamma) + beta
        
        return output.view(-1, seq_len, hidden_size)


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr, final_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr0 = 0.0
            lr1 = self.final_lr
        else:
            lr0 = self.final_lr
            lr1 = 0.0
        
        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i in [0, 1]:
                lrs.append(lr0)
            elif i in [2, 3]:
                lrs.append(lr1)
            else:
                raise ValueError("Optimizer should have exactly 2 parameter groups.")
        
        return lrs

