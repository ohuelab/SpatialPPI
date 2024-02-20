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


def rotation3D(tensor):
    aug_arg = random.sample(AUG_PARA, 1)[0]
    argus = [int(aug_arg[i]) for i in range(len(aug_arg))]
    tensor = np.rot90(tensor, k=argus[0], axes=(argus[1], argus[2]))
    tensor = np.rot90(tensor, k=argus[3], axes=(argus[4], argus[5]))
    return tensor


def relocation(data):
    dx = random.randint(-6, 6)
    dy = random.randint(-6, 6)
    dz = random.randint(-6, 6)
    padding = 6
    alen = data.shape[0]

    new_data = np.zeros([alen+padding*2, alen+padding*2, alen+padding*2, data.shape[3]], dtype=data.dtype)
    new_data[padding:padding+alen, padding:padding+alen, padding:padding+alen, :] = data
    new_data = new_data[padding + dx: padding + alen + dx, padding + dy: padding + alen + dy, padding + dz: padding + alen + dz, :]

    return new_data
