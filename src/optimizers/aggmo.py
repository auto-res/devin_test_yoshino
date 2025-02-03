import torch
from torch.optim.optimizer import Optimizer, required

class AggMo(Optimizer):
    def __init__(self, params, lr=required, betas=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            total_mom = float(len(betas))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    for beta in betas:
                        param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)
                for beta in betas:
                    buf = param_state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group['lr'] / total_mom, buf)
        return loss
