from .resnet3d import Resnet3DBuilder
from .densenet3d import DenseNet3DImageNet121, DenseNet3DPPI, DenseNet3D


def getModel(modelName, inputsize):
    if modelName == 'Resnet3D':
        resnet3D = Resnet3DBuilder.build_resnet_18(inputsize, 2)
        return resnet3D

    elif modelName == 'DenseNet3D':
        return DenseNet3DImageNet121(input_shape=inputsize,
                                     bottleneck=True,
                                     reduction=0.5,
                                     dropout_rate=0.2,
                                     weight_decay=1e-4,
                                     include_top=True,
                                     input_tensor=None,
                                     pooling='max',
                                     classes=2,
                                     activation='softmax')
    elif modelName == 'DenseNet3DPPI':
        return DenseNet3DPPI(input_shape=inputsize,
                             bottleneck=True,
                             reduction=0.5,
                             dropout_rate=0.2,
                             weight_decay=1e-4,
                             include_top=True,
                             input_tensor=None,
                             pooling='max',
                             classes=2,
                             activation='softmax')
    
    return DenseNet3D(
        inputsize,
        classes=2,
        activation='softmax'
    )
