import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_classifier(model_name, num_classes=4, **kwargs):
    # Get classifier model
    if model_name == 'resnet18':
        base_model = resnet18(**kwargs)
    elif model_name == 'resnet34':
        base_model = resnet34(**kwargs)
    elif model_name == 'resnet50':
        base_model = resnet50(**kwargs)
    elif model_name == 'resnet101':
        base_model = resnet101(**kwargs)
    elif model_name == 'resnet152':
        base_model = resnet152(**kwargs)
    else:
        print('Model name {} is not implemented'.format(model_name))
        raise TypeError

    # Num classes setting
    if "resnet" in model_name:
        base_model.fc = nn.Linear(base_model.fc.weight.shape[-1], num_classes)
    else:
        print('Model name {} is not implemented'.format(model_name))
        raise TypeError

    return base_model
