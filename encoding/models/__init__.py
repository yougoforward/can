from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .deeplabv3plus import*
from .can import *
from .can2 import *
from .can3 import *
from .can4 import *
from .dpcan import *
from .dpcan2 import *
from .dpcan3 import *
from .new_dpcan import *
from .new_can import *
from .new_can2 import *
from .new_can3 import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplabv3': get_deeplabv3,
        'deeplabv3plus': get_deeplabv3plus,
        'can': get_can,
        'can2': get_can2,
        'can3': get_can3,
        'can4': get_can4,
        'dpcan': get_dpcan,
        'dpcan2': get_dpcan2,
        'dpcan3': get_dpcan3,
        'new_dpcan': get_new_dpcan,
        'new_can': get_new_can,
        'new_can2': get_new_can2,
        'new_can3': get_new_can3,

    }
    return models[name.lower()](**kwargs)
