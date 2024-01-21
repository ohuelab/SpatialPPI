from .resnet3d import Resnet3DBuilder
from .densenet3d import DenseNet3DImageNet121, DenseNet3DPPI, DenseNet3D


def getModel(modelName, inputsize):
    if modelName == 'Resnet3D':
        resnet3D = Resnet3DBuilder.build_resnet_50(inputsize, 2)
        return resnet3D

    elif modelName == 'DenseNet3D':
        return DenseNet3D(
            inputsize,
            nb_dense_block=4,
            growth_rate=16,
            nb_filter=32,
            nb_layers_per_block=[4, 4, 4, 4],
            bottleneck=True,
            reduction=0.5,
            dropout_rate=0.2,
            weight_decay=1e-4,
            subsample_initial_block=True,
            include_top=True,
            input_tensor=None,
            pooling='max',
            classes=2,
            activation='softmax'
        )
