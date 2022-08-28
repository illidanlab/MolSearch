# MolSearch: Search-based Multi-objective Molecular Generation and Property Optimization

This is the official code repository of MolSearch.  

## Paper Abstract
Leveraging computational methods to generate small molecules with desired properties has been an active research area in the drug discovery field. Towards real-world applications, however, efficient generation of molecules that satisfy multiple property requirements simultaneously remains a key challenge. In this paper, we tackle this challenge using a search-based approach and propose a simple yet effective framework called MolSearch for multi-objective molecular generation (optimization). We show that given proper design and su"cient domain information, search-based methods can achieve performance comparable or even better than deep learning methods while being computationally efficient. Such efficiency enables massive exploration of chemical space given constrained computational resources. In particular, MolSearch starts with existing molecules and uses a two-stage search strategy to gradually modify them into new ones, based on transformation rules derived systematically and exhaustively from large compound libraries. We evaluate MolSearch in multiple benchmark generation settings and demonstrate its effectiveness and efficiency.

The current version of the paper can be found HERE (TBD).

## MCTS Environment Setup
```
conda create --name mcts python=3.7 pip 
conda install -c conda-forge rdkit
pip install pandas seaborn pickle-mixin
pip install -U scikit-learn==0.21.3 (RF scorer of GSK3B and JNK3 requires this version of sklearn)
```
## Design Move Transformations
Download the tranformation rules from [here](https://figshare.com/articles/dataset/chemblDB3_sqlitdb/12912080), whose link is provided by the design move github [page](https://github.com/mahendra-awale/medchem_moves). The transformation rules are derived from ChEMBL database at radius 3. Once the download is successful, put the downloaded file chemblDB3.sqlitdb (1.5GB) under MCTS folder. 

## Outside of MCTS folder
```
git clone https://github.com/mahendra-awale/medchem_moves
cd medchem_moves
python setup.py install
```
This step is to setup the design move transformation API, from the author github [page](https://github.com/mahendra-awale/medchem_moves).

## Running Commnands

stage 1

```
python frag_mcts_mo.py --goal gsk3b_jnk3 --start_mols task1 --max_child 5 --num_sims 20 --mol_idx 0 --seed 0 --scalar 0.7
```

stage2

```
python frag_mcts_mo_stage2.py --goal gsk3b_jnk3 --constraint gsk3b_jnk3 --start_mols task1 --max_child 3 --num_sims 10 --scalar 0.7 --group_idx 0 --seed 0 
```

# Acknowledgement

This research is funded in part by NSF IIS-1749940 (JZ), ONR N00014-20-1-2382 (JZ), NIH R01GM134307 (JZ, BC). This work is also supported via computational resources and services provided by the Institute for Cyber-Enabled Research ([ICER](https://icer.msu.edu/)) at MSU.