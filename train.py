import math
import os
import random
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.augmentation import augmentation
from utils.structure import getModel
from utils.dataset import gen, get_dataset_from_csv
from utils.record import saveConfig, drawFig


print(tf.test.gpu_device_name())
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(args):
    # load dataset
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print('---------------loading dataset---------------')
    input_size = [args.alength, args.alength, args.alength, args.ndims]
    train_set = get_dataset_from_csv(args.train_set, args.datapath)
    val_set = get_dataset_from_csv(args.val_set, args.datapath)

    print('dataset loaded, train set size', len(train_set[0]), 'test set size', len(val_set[0]))

    print('---------------augmentation---------------')
    train_set = augmentation(train_set, args.augment)
    val_set = augmentation(val_set, 0)
    print('dataset argumentation, train set size', len(train_set))

    print('---------------training---------------')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    savingPath = os.path.join(args.savingPath, timestamp)
    print("output path:", savingPath)

    if not os.path.exists(savingPath):
        os.makedirs(savingPath)

    saveConfig(savingPath, args)

    examine = 'accuracy'  # binary_accuracy accuracy
    monitor = f'val_{examine}'

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=10, verbose=1)

    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(savingPath, "best_model.hdf5"), monitor=monitor, verbose=1, save_best_only=True)
    save_latest = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(savingPath, "latest_model.hdf5"), monitor=monitor, verbose=1, save_best_only=False)

    model = getModel(args.model, input_size)

    if args.weights:
        print("Using trained weights from ")
        model.load_weights(args.weights)

    # model.summary()
    lossfun = keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss=lossfun,
                metrics=[examine, tf.keras.metrics.AUC()])


    history = model.fit_generator(generator=gen(train_set, args.batch, shuffle=True),
                                epochs=args.epoch,
                                steps_per_epoch=math.ceil(len(train_set)/args.batch),
                                validation_data=gen(val_set, args.batch, shuffle=False),
                                validation_steps=math.ceil(len(val_set)/args.batch),
                                callbacks=[save_best, earlystopper, save_latest],
                                workers=1,
                                verbose=1
                                )

    drawFig(history, examine, monitor, savingPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Resnet3D', 'DenseNet3D'], default='DenseNet3D', help='Backbone of the network')
    parser.add_argument('--datapath', type=str, default='./data/example_dataset/distance', help='Path to tensors')
    parser.add_argument('--weights', type=str, default='', help='Continue training based on weights')
    parser.add_argument('--savingPath', type=str, default='./models/example_model', help='Path to save models')

    parser.add_argument('--train_set', type=str, default='./data/example_dataset/part_0_train.csv', help='Path to train set csv file')
    parser.add_argument('--val_set', type=str, default='./data/example_dataset/part_0_val.csv', help='Path to val set csv file')

    parser.add_argument('--augment', type=int, default=24, help='value from 0-24 for argumentation, 0 for off')
    parser.add_argument('--batch', default=32)
    parser.add_argument('--alength', default=64, help='Tensor side length')
    parser.add_argument('--ndims', default=8, help='Tensor dims')
    parser.add_argument('--seed', default=2032, help='Random seeds')
    parser.add_argument('--epoch', default=40, help='Epoch to train')
    parser.add_argument('--lr', default=1e-5, help='Init learning rate')
    args, unknown = parser.parse_known_args()
    main(args)
