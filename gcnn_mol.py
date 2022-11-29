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

    m = gcnn.GCNN(
        train_args,
        dataset.atom_features_vector_length,
        train_args.number_of_molecules,
        dataset.mol_features_vector_length,
    )

    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()

    pbar = tqdm(range(100))

    # dataset targets, used for computing kendall's tau
    targets = [dp.target for dp in dataset]

    for epoch in pbar:
        running_loss = 0.0

        # save model outputs, used for computing kendall's tau
        outputs = []

        for i, datapoint in tqdm(
            enumerate(dataset.datapoints), desc="Datapoints:", leave=False, total=len(dataset.datapoints)
        ):
            target = datapoint.target

            # zero the parameter gradients
            optimizer.zero_grad()

            output = m.forward(datapoint)
            outputs.append(output.detach().numpy())

            loss = criterion(output, torch.Tensor([target]))
            loss.backward()

            running_loss += loss.item()

            optimizer.step()

        avg_loss = running_loss / len(dataset.datapoints)
        tau = kendalltau(targets, outputs)
        spearman = spearmanr(targets, outputs)
        pbar.set_description(
            f"Running loss: {running_loss}; Avg. Loss: {avg_loss}; Kendall's Tau: {tau.correlation}; Spearman: {spearman.correlation}"
        )
