import math
import torch
from torch.optim import Optimizer


class NewOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, betas_aggmo=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon,
                       betas_aggmo=betas_aggmo, step=0, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['momentum_buffers'] = {}
                    for beta in group['betas_aggmo']:
                        state['momentum_buffers'][beta] = torch.zeros_like(p.data)

                state['step'] += 1

                beta1, beta2 = group['beta1'], group['beta2']
                m, v = state['m'], state['v']
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step'])
                step_size /= (1 - beta1 ** state['step'])

                total_mom = float(len(group['betas_aggmo']))
                aggmo_update = torch.zeros_like(p.data)
                
                for beta in group['betas_aggmo']:
                    buf = state['momentum_buffers'][beta]
                    buf.mul_(beta).add_(grad)
                    aggmo_update.add_(buf)

                adam_update = step_size * m / (v.sqrt() + group['epsilon'])
                p.data.sub_(adam_update + group['lr'] * aggmo_update / total_mom)

        return loss
