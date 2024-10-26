import torch


class SODA(torch.optim.Optimizer):
    def __init__(self, params, rho=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SODA, self).__init__(params, defaults)
        
    @torch.no_grad()
    def perturb(self):   
        self.first_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (self.first_grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                param_state['e_w'] = e_w.clone()
        
    @torch.no_grad()
    def unperturb(self, flush=False):   
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                if 'e_w' not in param_state:
                    param_state['e_w'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                p.sub_(param_state['e_w'])  # get back to "w" from "w + e(w)"
                
                if flush:
                    param_state['e_w'].zero_()  # Clear the perturbation if flush is True
                    del param_state['e_w']

    @torch.no_grad()
    def step(self, zero_grad=False):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self, by=None):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if by is None:
            norm = torch.norm(
                        torch.stack([
                            p.grad.norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
        else:
            norm = torch.norm(
                        torch.stack([
                            self.state[p][by].norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
        
    @torch.no_grad()
    def _weight_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
