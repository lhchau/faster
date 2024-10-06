from .resnet import *
from .wideresnet import *
from .squeezenet import *
from .densenet import *
from torchvision.models import efficientnet_b0
import torchvision


def get_model(model_name, num_classes, widen_factor=1):
    if model_name == "resnet18":
        return resnet18(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet34":
        return resnet34(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet101":
        return resnet101(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet152":
        return resnet152(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "wideresnet28_10":
        return wideresnet28_10(num_classes=num_classes)
    elif model_name == "wideresnet40_2":
        return wideresnet40_2(num_classes=num_classes)
    elif model_name == "densenet121":
        return densenet121(num_classes=num_classes)
    elif model_name == "densenet169":
        return densenet169(num_classes=num_classes)
    elif model_name == "squeezenet":
        return squeezenet(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return efficientnet_b0(num_classes=num_classes)
    else:
        raise ValueError("Invalid model!!!")