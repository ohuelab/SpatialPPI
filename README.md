# SpatialPPI: Three-dimensional Space Protein-Protein Interaction Prediction With AlphaFold Multimer
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

![graphicalabstract](assets/ga.png)

## How to use

### Install environment

Using [Conda](https://www.anaconda.com/):

```bash
conda create -n spatial-ppi python=3.9.12
conda activate spatial-ppi
conda install tensorflow==2.9.1 scikit-learn==1.1.1 numpy pandas matplotlib
conda install -c conda-forge -c bioconda mmseqs2
pip install biopython==1.79 tqdm
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

### Prepare dataset

1. Prepare a dataset in `json` format, [example](https://github.com/ohuelab/SpatialPPI/blob/main/data/example_dataset.json)
2. Perform Alphafold Multimer prediction
3. Run `preprocess.py` to generate tensors and data list files

```bash
python preprocess.py \
--dataset [PATH to the json dataset] \
--data_dir [PATH to Alphafold Multimer prediction result folder] \
--work_dir [PATH to folder to save tensors and data files] \
--tensor_method [Methods for tensorization, choices: onehot, volume, distance, all] \
--split \ # Generate splited train, val and test dataset for 5-fold cross validation
--relaxed \ # Use relaxed Alphafold Multimer predictions
--threads [Number of threads to run] \
--models_per_pair [Number of Alphafold multimer models generated per protein pair]
```



### Run SpatialPPI for training

Run `train.py` to train models

```shell
python train.py \
--model [Backbone model to use] \
--datapath [PATH to data tensors] \
--weights [PATH to weights to fine-tuning] \
--savingPath [PATH to save trained models] \
--train_set [PATH to train set csv file generated in preprocess] \
--test_set [PATH to validation set csv file generated in preprocess]
```



### Run Spatial PPI for testing

`test.py` could use trained model to make predictions and evaluations

```shell
python test.py \
--model [Backbone model to use] \
--datapath [PATH to data tensors] \
--weights [PATH to weights to fine-tuning] \
--output [PATH to save prediction results] \
--test_set [PATH to test set csv file generated in preprocess]
```


## Resources
The data splits and weights trained from 5-fold cross-validation can be download from [here](https://drive.google.com/file/d/1ovLlK9zz3DYVnphCe_dwRjau_cwxRsUV/view?usp=sharing).

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Citation
- Hu W, Ohue M. **SpatialPPI: three-dimensional space protein-protein interaction prediction with AlphaFold Multimer**. _Computational and Structural Biotechnology Journal_, 2024. [doi: 10.1016/j.csbj.2024.03.009](https://doi.org/10.1016/j.csbj.2024.03.009)
