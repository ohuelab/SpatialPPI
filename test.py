import math
import os
import random
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from sklearn import metrics

from utils.structure import getModel
from utils.dataset import gen, get_dataset_from_csv


print(tf.test.gpu_device_name())
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print('---------------loading dataset---------------')
    input_size = [args.alength, args.alength, args.alength, args.ndims]
    test_set = get_dataset_from_csv(args.test_set, args.datapath)
    print('---------------test---------------')

    model = getModel(args.model, input_size)

    print("Using trained weights from", args.weights)
    model.load_weights(args.weights)
    # model.summary()

    model.compile(metrics=['accuracy', tf.keras.metrics.AUC()])

    pred = model.predict_generator(generator=gen(test_set, args.batch, shuffle=False, augment=False),
                                     steps=math.ceil(len(test_set) / args.batch),
                                      workers=1,
                                      verbose=1)

    pred = np.array(pred)
    np.save(args.output, pred)

    pred_score = np.array([0.5 - i[0] / 2 if i[0] > i[1] else i[1] / 2 + 0.5 for i in pred])

    label = np.array([float(i[1]) for i in test_set])

    tn, fp, fn, tp = confusion_matrix(label, pred_score.round(0)).ravel()
    fpr, tpr, thresholds_keras = roc_curve(label, pred_score)
    aucc = metrics.auc(fpr, tpr)

    print("accuracy:", round(accuracy_score(label, pred_score.round(0)), 4))
    print("AUC:", round(aucc, 4))
    print("precision:", round(tp / (tp + fp), 4))
    print("recall:", round(tp / (tp + fn), 4))
    print("acc | Positive", round(tp / (len(label) / 2), 4))
    print("acc | Negative", round(tn / (len(label) / 2), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Resnet3D', 'DenseNet3D'], default='DenseNet3D', help='Backbone of the network')
    parser.add_argument('--datapath', type=str, default='./data/example_dataset/distance', help='Path to tensors')
    parser.add_argument('--weights', type=str,
                        default='./models/example_model/20231111-090154/best_model.hdf5')
    parser.add_argument('--output', type=str, default='./models/example_model/20231111-090154/preds.npy')
    parser.add_argument('--test_set', type=str, default='./data/example_dataset/part_0_test.csv')

    parser.add_argument('--batch', default=32)
    parser.add_argument('--alength', default=64)
    parser.add_argument('--ndims', default=8)
    parser.add_argument('--seed', default=2032)
    args, unknown = parser.parse_known_args()

    main(args)
