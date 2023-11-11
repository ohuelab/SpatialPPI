import numpy as np
import pandas as pd
import json


PDBFS = {
    'ATOM': [0, 4],
    'atype': [13, 16],
    'resName': [17, 20],
    'CHAIN': 21,
    'resSeq': [22, 26],
    'X': [30, 38],
    'Y': [38, 46],
    'Z': [46, 54],
    'plddt': [61, 66],
    'ELEMENT': 77
}


def extractPDB(row, target):
    if target in ('ATOM', 'resName', 'atype'):
        return row[PDBFS[target][0]:PDBFS[target][1]].replace(' ', '')
    elif target in ('X', 'Y', 'Z', 'plddt'):
        return float(row[PDBFS[target][0]:PDBFS[target][1]])
    elif target in ('ELEMENT', 'CHAIN'):
        return row[PDBFS[target]]
    elif target in ('resSeq'):
        return int(row[PDBFS[target][0]:PDBFS[target][1]])


def readPDB(PDBfilename):
    file = open(PDBfilename, 'r', encoding="utf-8")
    PDBfile = [i.replace('\n', '') for i in file.readlines()]
    file.close()
    pdb = []
    for i in PDBfile:
        if extractPDB(i, 'ATOM') == 'ATOM':
            pdb.append({j: extractPDB(i, j) for j in PDBFS.keys()})
    return pdb


def readPDB2Pd(PDBfilename):
    file = open(PDBfilename, 'r', encoding="utf-8")
    PDBfile = [i.replace('\n', '') for i in file.readlines()]
    file.close()
    pdb = {i:[] for i in list(PDBFS.keys())[1:]}

    for i in PDBfile:
        if extractPDB(i, 'ATOM') == 'ATOM':
            for j in list(PDBFS.keys())[1:]:
                pdb[j].append(extractPDB(i, j))
    
    return pd.DataFrame(pdb)


def filelist():
    with open('./../02_AlphafoldMultimerGeneration/datasets.txt', 'r', encoding="utf-8") as f:
        success = f.readlines()
        success = [i.split('.')[0] for i in success]

    filelists = []

    for i in success:
        for j in range(5):
            filelists.append(f'./../02_AlphafoldMultimerGeneration/models/{i}/ranked_{j}.pdb')

    return filelists


def filelistByName():
    with open('./../02_AlphafoldMultimerGeneration/datasets.txt', 'r', encoding="utf-8") as f:
        success = f.readlines()
        success = [i.split('.')[0] for i in success]

    return success


def filelistFromJSON():
    with open("./../02_AlphafoldMultimerGeneration/dataset1200.json", 'r', encoding="utf-8") as f:
        dataset = json.loads(f.read())
        dataset = [fastaname(i) for i in dataset]

    return dataset


def getLabels():
    with open('./../02_AlphafoldMultimerGeneration/datalabels.txt', 'r', encoding="utf-8") as f:
        label = f.readlines()
    label = [i.replace('\n', '').split(' ') for i in label]
    return label


def fastaname(data):
    return f"{data['interactors'][0]['UniProt Entry']}-{data['interactors'][1]['UniProt Entry']}"


def geneLabels():
    with open("./../02_AlphafoldMultimerGeneration/dataset606.json", 'r', encoding="utf-8") as f:
        dataset606 = json.loads(f.read())
    dataset = {fastaname(i): i['type'] for i in dataset606}

    with open('./../02_AlphafoldMultimerGeneration/datasets.txt', 'r', encoding="utf-8") as f:
        success = f.readlines()
        success = [i.split('.')[0] for i in success]

    filelists = []

    for i in success:
        for j in range(5):
            filelists.append(f'./../02_AlphafoldMultimerGeneration/models/{i}/ranked_{j}.pdb {dataset[i]}\n')

    with open('./../02_AlphafoldMultimerGeneration/datalabels.txt', 'w', encoding="utf-8") as f:
        f.writelines(filelists)

    print('Done')


def getDataset(path):
    with open(path, 'r', encoding="utf-8") as f:
        dataset = json.loads(f.read())
    return dataset


def writeFasta(dataset, dir):
    for i in dataset:
        item = {
            'name': f"{i['interactors'][0]['UniProt Entry']}-{i['interactors'][1]['UniProt Entry']}.fasta",
            'proteinA': i['interactors'][0]['UniProt Entry'],
            'proteinB': i['interactors'][1]['UniProt Entry'],
            'sequenceA': i['interactors'][0]['sequence'],
            'sequenceB': i['interactors'][1]['sequence']
        }
        path = dir + item['name']
        with open(path, 'w', encoding="utf-8") as f:
            f.write(f">{item['proteinA']}\n")
            f.write(item['sequenceA'] + '\n')
            f.write(f">{item['proteinB']}\n")
            f.write(item['sequenceB'] + '\n')
