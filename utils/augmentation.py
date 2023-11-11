import json
import numpy as np
import random

AUG_PARA = ['001001',
            '001101',
            '001201',
            '001301',
            '102001',
            '102101',
            '102201',
            '102301',
            '202001',
            '202101',
            '202201',
            '202301',
            '302001',
            '302101',
            '302201',
            '302301',
            '112001',
            '112101',
            '112201',
            '112301',
            '312001',
            '312101',
            '312201',
            '312301']


def rotation3D(tensor, args):
    if len(args):
        argus = [int(args[i]) for i in range(len(args))]
        tensor = np.rot90(tensor, k=argus[0], axes=(argus[1], argus[2]))
        tensor = np.rot90(tensor, k=argus[3], axes=(argus[4], argus[5]))
    return tensor


def augmentation(ds, argument=1):
    data = []
    for i in range(len(ds[0])):
        if argument:
            args = random.sample(AUG_PARA, argument)
        else:
            args = ['']
        for a in args:
            data.append([ds[0][i], ds[1][i], a])
    if argument:
        random.shuffle(data)
    return data
