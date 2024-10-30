from torch.optim import SGD, Adam
from .sam import SAM

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperpara={}):
    if opt_name == 'sam':
        return SAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sgd':
        return SGD(net.parameters(), **opt_hyperpara)
    elif opt_name == 'adam':
        return Adam(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")