from rdkit import Chem
from args import TrainArgs
from data.featurizer import MoleculeFeaturizer
from data.dataset import load_dataset
from torch.utils.data import DataLoader
from model import gcnn
from torch import nn
from tqdm.auto import tqdm

import torch
import torch.optim as optim

if __name__ == "__main__":
    train_args = TrainArgs().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(train_args)
    dataset.to(device)

    scalers = dataset.fit_scalers_to_features()
    dataset.normalize_features(scalers)

    m = gcnn.GCNN(
        train_args,
        dataset.atom_feature_vector_length,
        train_args.number_of_molecules,
        dataset.mol_feature_vector_length,
    )

    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()

    pbar = tqdm(range(100))

    for epoch in pbar:
        running_loss = 0.0
        for i, datapoint in enumerate(dataset.datapoints):
            target = datapoint.target

            # zero the parameter gradients
            optimizer.zero_grad()

            output = m.forward(datapoint)

            loss = criterion(output, torch.Tensor([target]))
            loss.backward()

            running_loss += loss.item()

            optimizer.step()

        pbar.set_description(f"Running loss: {running_loss}")
