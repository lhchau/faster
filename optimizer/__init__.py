from torch.optim import SGD, Adam
from .sam import SAM
from .faster import FASTER
from .samadam import SAMADAM
from .soda import SODA

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperpara={}):
    if opt_name == 'sam':
        return SAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sgd':
        return SGD(net.parameters(), **opt_hyperpara)
    elif opt_name == 'soda':
        return SODA(net.parameters(), **opt_hyperpara)
    elif opt_name == 'adam':
        return Adam(net.parameters(), **opt_hyperpara)
    elif opt_name == 'faster':
        return FASTER(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samadam':
        return SAMADAM(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")