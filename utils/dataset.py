import os
import csv
import random
import numpy as np
import tensorflow as tf

from .augmentation import rotation3D


def gen(dataset, batch_size, shuffle):
    while 1:
        if shuffle:
            random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            ds = dataset[i:i + batch_size]
            x = []
            y = []
            for d in ds:
                data = np.load(d[0])
                if len(d) > 2:
                    data = rotation3D(data, d[2])
                label = [0, 1] if int(d[1]) == 1 else [1, 0]

                x.append(data)
                y.append(label)

            x = np.array(x)
            y = np.array(y)
            yield [x, y]


def get_dataset_from_csv(csv_path, data_dir):
    x = []
    y = []
    with open(csv_path) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            x.append(os.path.join(data_dir, row[0]))
            y.append(row[1])
    return [x, y]


def save_dataset_to_csv(csv_path, dataset, models_per_pair=5):
    contents = []
    for item in dataset:
        for i in range(models_per_pair):
            label = 1 if item['type'] == 'positive' else 0
            name = item['name'] + f"-{i}.npy"
            contents.append([name, label])

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(contents)
