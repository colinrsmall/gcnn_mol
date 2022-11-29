# gcnn_mol

gcnn_mol is a project developing graph convolutional neural networks for molecule property prediction. Special
emphasis is placed on multi-molecular inputs (e.g. pairs of molecules with a single numerical target) and data 
interpretability.

Currently, gcnn_mol is being developed for the prediction of antibiotic synergy. In this project, we want to train
a graph neural network on datasets of paired antibiotic Bliss scores to predict novel, potentially synergistic
pairs of antibiotics.

## TODO:

- [x] ~~Generate molecular features~~ Done 11/29
- [ ] Move training code to train.py
- [ ] Incorporate logging with Neptune
- [ ] Training/test split during training
- [ ] Implement automated hyperparameter optimization
- [ ] Fix target value normalization
- [ ] Create predict.py script to run
- [ ] Implement more activation functions:
  - [ ] PReLu
  - [ ] SeLu
  - [ ] TanH
  - [ ] Leaky ReLu
- [ ] Implement additional loss functions
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
