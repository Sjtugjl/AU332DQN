# -*- coding:utf-8 -*-
import os
import sys
import random
import numpy as np

# set random seed. random seeds other than 0 are also okay.
def set_random_seed(i = 0,cyf=1):
    # python hash seed
    os.environ['PYTHONHASHSEED'] = str(i)
    # if you use torch
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
    # if you use tensorflow
    try:
        import tensorflow
    except ImportError:
        pass
    else:
        if cyf:
            tensorflow.set_random_seed(i)
        else:
            tensorflow.random.set_seed(i)
    # numpy
    np.random.seed(i)
    # python random lib
    random.seed(i)
    ########## YOUR CODE BEGIN ##########
    # If you use other libraries,
    # please set the random seed to get reproduceable result.
    
    ##########  YOUR CODE END  ##########