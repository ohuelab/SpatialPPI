import os
import json
import csv
import pandas as pd
import numpy as np
import random

from .readPDB import getDataset, fastaname



def save_dataset_to_csv(csv_path, data, models_per_pair=5, use_ranking=False, data_dir=''):
    contents = []
    for index, row in data.iterrows():
        label = 1 if row['type'] == 'positive' else 0
        if use_ranking:
            ranking_path = os.path.join(data_dir, index, "ranking_debug.json")
            if os.path.exists(ranking_path):
                with open(ranking_path, 'r') as f:
                    ranking = json.load(f)
                    idx = int(ranking['order'][0].split('_')[1])-1
            else:
                idx = 0
            name = index + f"-{idx}.npy"
            contents.append([name, label])
        else:
            for i in range(models_per_pair):
                name = index + f"-{i}.npy"
                contents.append([name, label])

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(contents)


def split_by_cluster(dataset, work_dir, n_split, models_per_pair=5, data_dir=''):
    assert len(dataset) % (n_split * 2) == 0, f'Can not split to {n_split} parts'

    fasta_path = os.path.join(work_dir, 'DB.fasta')
    db_path = os.path.join(work_dir, 'DB')

    with open(fasta_path, 'w') as f:
        for rec in dataset:
            f.write(f">{fastaname(rec)}\n")
            f.write(f"{rec['interactors'][0]['sequence']}{rec['interactors'][1]['sequence']}\n")

    os.system(f"mmseqs easy-cluster {fasta_path} {os.path.join(work_dir, 'clusterRes')} {os.path.join(work_dir, 'tmp')}")
    
    cluster = pd.read_csv(os.path.join(work_dir, 'clusterRes_cluster.tsv'), sep='\t', header=None)
    grouped = cluster.groupby(0)
    clusters = []
    for name,group in grouped:
        clusters.append(group[1].to_list())
    
    allocation = {}
    for i in range(len(dataset)):
        item = dataset[i]
        allocation[fastaname(item)] = {'idx': i, 'type': item['type'], 'label': -1}
    allocation = pd.DataFrame(allocation).T

    for c in clusters:
        if len(c) != 1:
            idx = random.randint(0, n_split-1)
            for k in c:
                allocation.loc[k, 'label'] = idx
    
    recPerPart = int(len(dataset) / (n_split * 2))
    for i in range(n_split):
        _, counts = np.unique(allocation[allocation['label'] == i]['type'].to_numpy(), return_counts=True)
        assert counts[0] <= recPerPart
        assert counts[1] <= recPerPart
        types = ['negative', 'positive']
        for j in range(2):
            slice = allocation[(allocation['label'] == -1) & (allocation['type'] == types[j])].sample(recPerPart - counts[j])
            allocation.loc[slice.index, 'label'] = i

    for idx in range(5):
        train = []
        test = []

        test = allocation[allocation['label'] == idx]
        train = allocation[allocation['label'] != idx]
        
        save_dataset_to_csv(os.path.join(work_dir, f'part_{idx}_train.csv'), train, models_per_pair, False, '')
        save_dataset_to_csv(os.path.join(work_dir, f'part_{idx}_test.csv'), test, 1, (models_per_pair != 1), data_dir)
    