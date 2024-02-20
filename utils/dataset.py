import os
import csv
import random
import numpy as np
import tensorflow as tf

from .augmentation import rotation3D, relocation


def gen(dataset, batch_size, shuffle, augment=False):
    while 1:
        if shuffle:
            random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            ds = dataset[i:i + batch_size]
            x = []
            y = []
            for d in ds:
                data = np.load(d[0])
                if augment:
                    data = rotation3D(data)
                    data = relocation(data)

                label = [0, 1] if int(d[1]) == 1 else [1, 0]

                x.append(data)
                y.append(label)

            x = np.array(x)
            y = np.array(y)
            yield [x, y]


def get_dataset_from_csv(csv_path, data_dir, shuffle=False):
    data = []
    with open(csv_path) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            x = os.path.join(data_dir, row[0])
            y = row[1]
            data.append([x, y])

    if shuffle:
            random.shuffle(data)
    return data


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
