# gcnn_mol

gcnn_mol is a project developing graph convolutional neural networks for molecule property prediction. Special
emphasis is placed on multi-molecular inputs (e.g. pairs of molecules with a single numerical target) and data 
interpretability.

Currently, gcnn_mol is being developed for the prediction of antibiotic synergy. In this project, we want to train
a graph neural network on datasets of paired antibiotic Bliss scores to predict novel, potentially synergistic
pairs of antibiotics.

## TODO:

- [x] ~~Generate molecular features~~ Done 11/29
- [x] ~~Move training code to train.py~~ Done 11/29
- [x] ~~Incorporate logging with Neptune~~ Done 11/30 - Using wandb instead
- [x] ~~Training/test split during training~~ Done 11/29
- [x] Implement automated hyperparameter optimization
- [x] ~~Fix target value normalization~~ Done 11/29 - Decided not to implement target scaling
- [ ] Create predict.py script to run
- [x] ~~Implement more activation functions:~~ Done 12/6
  - [x] PReLu
  - [x] SeLu
  - [x] TanH
  - [x] Leaky ReLu
- [x] ~~Implement additional loss functions~~ Done 12/6 - Although I am considering adding margin-ranking loss too
- [x] Fix readout dropout from generating NaN
- [x] Implement more aggregation methods
  - [x] Max
  - [x] Min
  - [x] LAF https://arxiv.org/abs/2012.08482
- [x] Graph attention mechanism
- [ ] Change train/test split function to work differently for Kulesa et al. data (split data within each common antibiotic type so that e.g. 60% of all vancomycin, cyclosporine, etc. combinations are in the train set and 40% are in the test set)
- [ ] Motif/subgraph attention
- [ ] Co-attention
- [ ] Motif/subgraph co-attention
- [ ] Change atom features to those used in https://arxiv.org/pdf/1603.00856v3.pdf

Nice to have:

- [ ] Clean up data normalization code
- [ ] Jumping Knowledge Network: https://arxiv.org/abs/1806.03536
- [ ] Self-attention graph pooling: https://arxiv.org/abs/1904.08082
