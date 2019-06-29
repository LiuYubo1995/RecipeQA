# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import re
import torch.nn as nn


__all__ = ['ResNet50_2048']

model_urls = {
    'ResNet50_2048': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]


pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def load_pretrained(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model

class ResNet50_2048(nn.Module):
    def __init__(self, original_model = models.resnet50(pretrained=True)):
        super(ResNet50_2048, self).__init__()
        num_classes = 1000
        self.settings = pretrained_settings['ResNet50_2048']['imagenet']
        model = load_pretrained(original_model, num_classes, self.settings)
        self.features = nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x):
        x = self.features(x)
        return x



# res50_model = models.resnet50(pretrained=True)
# res50_conv2 = ResNet50Bottom(res50_model)
#
# outputs = res50_conv2(inputs)
# outputs.data.shape  # => torch.Size([4, 2048, 7, 7])

