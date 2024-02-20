import argparse
import os
import json
import random

from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils.interface import cal_interface_worker
from utils.encode import encoding_worker
from utils.dataset import save_dataset_to_csv
from utils.cluster import split_by_cluster


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    with open(args.dataset, 'r', encoding="utf-8") as f:
        dataset = json.loads(f.read())

    # prepare data
    jname = []
    labels = []
    pdb_paths = []
    for item in dataset:
        for i in range(args.models_per_pair):
            if args.relaxed:
                pdb_path = f"{args.data_dir}/{item['name']}/relaxed_model_{(i + 1)}_multimer_v3_pred_0.pdb"
            else:
                pdb_path = f"{args.data_dir}/{item['name']}/unrelaxed_model_{(i + 1)}_multimer_v3_pred_0.pdb"

            assert os.path.exists(pdb_path)

            labels.append(1 if item['type'] == 'positive' else 0)
            jname.append(item['name'] + f"-{i}")
            pdb_paths.append(pdb_path)


    # calculate interface
    interface_file = os.path.join(args.work_dir, 'interfaces.npy')
    if os.path.exists(interface_file) and not args.enforce_calculation:
        interfaces = np.load(interface_file, allow_pickle=True)
        print("interface file exists")
    else:
        print("calculate interface")
        with Pool(processes=args.threads) as pool:
            packed_data = [[pdb_paths[i],
                            args.interface_threshold,
                            args.interface_least_residue_n,
                            args.relaxed] for i in range(len(jname))]
            interfaces = list(tqdm(pool.imap(cal_interface_worker, packed_data), total=len(packed_data)))

        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
        np.save(interface_file, interfaces)

    # encoding
    print("encoding tensors")

    method_list = ['onehot', 'distance', 'volume']
    if args.tensor_method != 'all':
        method_list = [args.tensor_method]

    for method in method_list:
        print(f"processing {method} method")
        with Pool(processes=args.threads) as pool:
            packed_data = []
            outpath = os.path.join(args.work_dir, method)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            for i in range(len(jname)):
                packed_data.append([pdb_paths[i],
                                    interfaces[i],
                                    args.edge_length,
                                    method,
                                    True,
                                    args.interface_threshold,
                                    os.path.join(outpath, jname[i] + '.npy'),
                                    args.relaxed])
            result = list(tqdm(pool.imap_unordered(encoding_worker, packed_data), total=len(packed_data)))

    # split dataset
    if args.split:
        split_by_cluster(dataset, args.work_dir, 5, 5, args.data_dir)

    else:
        save_dataset_to_csv(os.path.join(args.work_dir, 'datalist.csv'), dataset, args.models_per_pair)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/dataset.json', help='JSON dataset')
    parser.add_argument('--data_dir', type=str, default='./data/af_models', help='Prefix to the Alphafold Multimer prediction result folder')
    parser.add_argument('--work_dir', type=str, default='./data/example_dataset', help='Output dir to save the tensors')
    parser.add_argument('--tensor_method', choices=['onehot', 'volume', 'distance', 'all'], default='distance')
    parser.add_argument('--interface_threshold', type=float, default=12.0, help='Distance threshold to define two residue contact')
    parser.add_argument('--interface_least_residue_n', type=int, default=12, help='Increase the distance threshold to ensure at least n residue included in the nearst area')
    parser.add_argument('--enforce_calculation', action='store_true', help='Enforce the calculation even if the file exists')
    parser.add_argument('--split', action='store_true', help='Generate splited train, val and test dataset for 5-fold cross validation')
    parser.add_argument('--relaxed', action='store_true', help='Use relaxed models')
    parser.add_argument('--edge_length', type=int, default=64, help='Edge length of the tensor')
    parser.add_argument('--threads', type=int, default=12, help='Number of threads running')
    parser.add_argument('--models_per_pair', type=int, default=5, help='Number of Alphafold multimer models generated per protein pair')
    parser.add_argument('--seed', default=2032, type=int, help='Random seeds')
    args, unknown = parser.parse_known_args()
    main(args)