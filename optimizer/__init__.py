from torch.optim import SGD, Adam
from .sam import SAM
from .leesin import LEESIN

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
    elif opt_name == 'leesin':
        return LEESIN(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")