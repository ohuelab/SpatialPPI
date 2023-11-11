import Bio.PDB
import json
import random
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN

from .readPDB import readPDB, readPDB2Pd


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"] - residue_two["CA"]
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def calculate_interface(pdb_path, threshold=12.0, relaxed=False):
    if relaxed:
        chainAchar = 'A'
        chainBchar = 'B'
    else:
        chainAchar = 'B'
        chainBchar = 'C'

    structure = Bio.PDB.PDBParser().get_structure("complex", pdb_path)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model[chainAchar], model[chainBchar])
    contact_map = dist_matrix < threshold

    interface_a = np.sum(contact_map, axis=1) > 0
    interface_b = np.sum(contact_map, axis=0) > 0

    pdb = readPDB2Pd(pdb_path)
    starterA = pdb[(pdb['CHAIN'] == chainAchar)]['resSeq'].min()
    starterB = pdb[(pdb['CHAIN'] == chainBchar)]['resSeq'].min()
    if starterA != 1:
        blank = [False for _ in range(starterA - 1)]
        interface_a = np.hstack((blank, interface_a))
    if starterB != 1:
        blank = [False for _ in range(starterB - 1)]
        interface_b = np.hstack((blank, interface_b))

    return [interface_a, interface_b]


def cal_interface_worker(arg):
    while True:
        interface = calculate_interface_plddt(arg[0], contact_threshold=arg[1], relaxed=arg[3])
        # interface_cleaned = remove_dissociate(arg[0], arg[1], interface)

        if (np.count_nonzero(interface[0]) + np.count_nonzero(interface[1])) > arg[2]:
            break
        elif arg[1] >= 30:
            break
        else:
            arg[1] += 4
    return interface


def remove_dissociate(pdb_path, plddt_path, interface):
    PDB_content = readPDB(pdb_path)
    F = open(plddt_path, 'rb')
    content = pickle.load(F)
    PLDDT = content['plddt']

    PDB_content = [i for i in PDB_content if i['ELEMENT'] == 'C']
    content = []
    for i in PDB_content:
        if i['ELEMENT'] != 'C':
            continue
        pindex = i['resSeq'] - 1 + (0 if i['CHAIN'] == 'A' else len(interface[0]))
        if PLDDT[pindex] < 40:
            continue
        if interface[0 if i['CHAIN'] == 'A' else 1][i['resSeq'] - 1]:
            content.append(i)

    if len(content) == 0:
        return interface
    coords = np.array([[item['X'], item['Y'], item['Z']] for item in content])
    label = DBSCAN(eps=12, min_samples=5).fit(coords)
    content = np.array(content)

    gp_idx = np.argmax([np.count_nonzero(label.labels_ == i) for i in set(label.labels_)])
    cleaned_interface = [np.full(len(interface[0]), False), np.full(len(interface[1]), False)]
    content = content[np.array(label.labels_) == gp_idx]
    for i in content:
        cleaned_interface[0 if i['CHAIN'] == 'A' else 1][i['resSeq'] - 1] = True

    return cleaned_interface


def cal_dist(residue_x, residue_y) :
    coord_x = residue_x.loc[:, ['X', 'Y', 'Z']].iloc[0].to_numpy()
    coord_y = residue_y.loc[:, ['X', 'Y', 'Z']].iloc[0].to_numpy()
    dist_vec = coord_x - coord_y
    return np.sqrt(np.sum(dist_vec*dist_vec))


def calculate_interface_plddt(pdb_path, plddt_threshold=50, contact_threshold=8.0, relaxed=True):
    if relaxed:
        chainAchar = 'A'
        chainBchar = 'B'
    else:
        chainAchar = 'B'
        chainBchar = 'C'

    pdb = readPDB2Pd(pdb_path)

    PLDDT_A = []
    PLDDT_B = []
    for i in range(len(pdb)):
        if i == 0 or pdb.iloc[i-1]['resSeq'] != pdb.iloc[i]['resSeq']:
            if pdb.iloc[i]['CHAIN'] == chainAchar:
                PLDDT_A.append(pdb.iloc[i]['plddt'])
            else:
                PLDDT_B.append(pdb.iloc[i]['plddt'])

    structure = Bio.PDB.PDBParser().get_structure("complex", pdb_path)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model[chainAchar], model[chainBchar])
    lenchainA = dist_matrix.shape[0]

    for i in range(len(PLDDT_A)):
        if PLDDT_A[i] < plddt_threshold:
            dist_matrix[i, ] = float('inf')
    for j in range(len(PLDDT_B)):
        if PLDDT_B[j] < plddt_threshold:
            dist_matrix[:, j] = float('inf')

    contact_map = dist_matrix < contact_threshold

    interface_a = np.sum(contact_map, axis=1) > 0
    interface_b = np.sum(contact_map, axis=0) > 0

    starterA = pdb[(pdb['CHAIN'] == chainAchar)]['resSeq'].min()
    starterB = pdb[(pdb['CHAIN'] == chainBchar)]['resSeq'].min()
    if starterA != 1:
        blank = [False for _ in range(starterA - 1)]
        interface_a = np.hstack((blank, interface_a))
    if starterB != 1:
        blank = [False for _ in range(starterB - 1)]
        interface_b = np.hstack((blank, interface_b))

    return [interface_a, interface_b]
