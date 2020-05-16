# EVERGREEN

## Installation
conda create -n evergreen python=3
conda activate evergreen
conda install numpy tensorflow-gpu=1.15 git h5py
pip install gym
pip install pygame
git clone https://github.com/adamjcalhoun/Evergreen.git

## Examples for running


## Models to test
Train on two different patch types - coarse, dense, and all pairwise combinations of small/large
Train several models - dense, LSTM
Also: odor memory?
Show chemotaxis ability
Show foraging
Show representations

### Model set 1
Dense network
-> 1 or 3 hidden layers
-> reward hunger or reward odor
-> coarse (patchy) and dense patches (dense patches should have equivalent amounts of food)
