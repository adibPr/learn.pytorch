#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

import torchvision

def progress_bar (*arg) :
    if len (arg) == 1 : 
        sys.stdout.flush()
        sys.stdout.write ('\n')
        return
    else : 
        idx, length = arg

    bar_len = 60
    filled_len = int(round(bar_len * idx / float(length)))

    percents = round(100.0 * idx / float(length), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s \r' % (bar, percents, '%'))
    sys.stdout.flush()

def imshow_tensor(img):
    if img.shape[0] != 3 : 
        # means its a dataset, make grid first
        img = torchvision.utils.make_grid (img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show ()
