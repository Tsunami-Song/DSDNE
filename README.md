# DSDNE(Directed Structural Deep Network Encoder)
A variation of SDNE that allows the model to learn from illicit nodes in Bitcoin trading graph..

## Installation
* Tested with Python 3.7 and Torch 1.11.0+cu113
* Clone the repository
```
https://github.com/Tsunami-Song/DSDNE.git
cd DSDNE
```
* Set up the environment (Anaconda): 
```
conda create -n LabelGCN python=3.7
conda activate DSDNE
pip install -r requirements.txt
```

## Datasets

The Elliptic dataset is available at https://www.kaggle.com/ellipticco/elliptic-data-set. This project expects the Elliptic dataset to be located under a directory named `elliptic_bitcoin_dataset`.

NOTE: All runs involving the Elliptic dataset are computationally demanding. Producing Tables 2 and 3 of the paper 
required running on a server for several hours.