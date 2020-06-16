from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pascal import PascalSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
# from .pcontext60 import ContextSegmentation
from .cocostuff import CocostuffSegmentation

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pascal': PascalSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes': CitySegmentation,
    'cocostuff': CocostuffSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
