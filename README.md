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
- [ ] Implement automated hyperparameter optimization
- [x] ~~Fix target value normalization~~ Done 11/29 - Decided not to implement target scaling
- [ ] Create predict.py script to run
- [x] ~~Implement more activation functions:~~ Done 12/6
  - [x] PReLu
  - [x] SeLu
  - [x] TanH
  - [x] Leaky ReLu
- [x] ~~Implement additional loss functions~~ Done 12/6 - Although I am considering adding margin-ranking loss too
- [ ] Fix readout dropout from generating NaN
- [ ] Implement more aggregation methods
  - [ ] Max
  - [ ] Min
  - [ ] LSTM (see SAGEConv in GraphSAGE)
- [ ] Graph attention mechanism
- [ ] Motif/subgraph attention
- [ ] Co-attention
- [ ] Motif/subgraph co-attention
- [ ] Add optimizers to arguments (Adam, SGD, etc.)

Nice to have:

- [ ] Clean up data normalization code
- [ ] Jumping Knowledge Network: https://arxiv.org/abs/1806.03536
- [ ] Self-attention graph pooling: https://arxiv.org/abs/1904.08082
