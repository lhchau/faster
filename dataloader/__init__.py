from .cifar100 import get_cifar100
from .cifar10 import get_cifar10
from .animal10n import get_animal10n
from .tiny_imagenet import get_tiny_imagenet

# Data
def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=4,
    data_augmentation="standard",
):
    print('==> Preparing data..')

    if data_name == "cifar100":
        return get_cifar100(batch_size, num_workers, data_augmentation)
    elif data_name == "cifar10":
        return get_cifar10(batch_size, num_workers, data_augmentation)
    elif data_name == "tiny_imagenet":
        return get_tiny_imagenet(batch_size, num_workers)
    elif data_name == "animal10n":
        return get_animal10n(batch_size, num_workers)