from rdkit import Chem
from args import TrainArgs
from data.featurizer import MoleculeFeaturizer
from data.dataset import load_dataset
from torch.utils.data import DataLoader
from model import gcnn
from torch import nn
from tqdm.auto import tqdm
import numpy as np

from scipy.stats import kendalltau, spearmanr
import torch
import torch.optim as optim

if __name__ == "__main__":
    train_args = TrainArgs().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(train_args)
    dataset.to(device)

    feature_scalers = dataset.fit_scalers_to_features()
    dataset.normalize_features(feature_scalers)

    # target_scaler = dataset.fit_scalers_to_target()
    # dataset.normalize_targets(target_scaler)

    if train_args.mol_features_only:
        m = gcnn.FCNNOnly(
            train_args,
            dataset.mol_features_vector_length,
            train_args.number_of_molecules,
        )
    else:
        m = gcnn.GCNN(
            train_args,
            dataset.atom_features_vector_length,
            train_args.number_of_molecules,
            dataset.mol_features_vector_length,
        )

    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()

    pbar = tqdm(range(100))

    train_set, test_set = dataset.train_test_split(0.2)

    for epoch in pbar:
        training_loss = 0.0

        # save model outputs, used for computing kendall's tau
        outputs = []

        for i, datapoint in tqdm(enumerate(train_set), desc="Train set:", leave=False, total=len(train_set)):
            target = datapoint.target

            # zero the parameter gradients
            optimizer.zero_grad()

            output = m.forward(datapoint)
            outputs.append(output.detach().numpy())

            loss = criterion(output, torch.Tensor([target]))
            loss.backward()

            training_loss += loss.item()

            optimizer.step()

        test_loss = 0
        test_outputs = []
        test_targets = [dp.target for dp in test_set]

        for datapoint in tqdm(test_set, desc="Test set:", leave=False, total=len(test_set)):
            target = datapoint.target
            output = m.forward(datapoint)
            test_outputs.append(output.detach().numpy())

            loss = criterion(output, torch.Tensor([target]))
            test_loss += loss.item()

        avg_loss = test_loss / len(test_set)
        tau = kendalltau(test_targets, test_outputs)
        spearman = spearmanr(test_targets, test_outputs)

        pbar.set_description(
            f"Test set: Avg. Loss: {avg_loss}; Kendall's Tau: {tau.correlation}; Spearman: {spearman.correlation}"
        )
