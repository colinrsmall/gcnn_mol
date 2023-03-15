import csv
import sys

from utils.args import TrainArgs
from data.dataset import load_dataset
from model import gcnn
from torch import nn
from tqdm.auto import tqdm
from utils import metrics, utils

import torch
import torch.optim as optim

import wandb
import yaml


def train_model(load_path: str, dataset_path: str, output_path: str):
    # Load saved model
    state_dict, atom_features_scalers, train_args = utils.load_checkpoint(load_path)
    train_args.dataset_path = dataset_path

    # Set pytorch seed
    if train_args.seed:
        torch.manual_seed(train_args.seed)

    # Detect device
    if train_args.cpu:
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data according to train_args settings
    dataset = load_dataset(train_args)

    # Scale dataset features
    dataset.normalize_features(atom_features_scalers)

    # Move dateset to detected device
    dataset.to(device)

    # Build model
    m = gcnn.GCNN(
        train_args,
        dataset.atom_features_vector_length,
        train_args.number_of_molecules,
        dataset.mol_features_vector_length,
        device,
    ).to(device)
    m.load_state_dict(state_dict)

    with open(output_path, "w+", newline="") as output_file:
        file_writer = csv.writer(output_file)
        if train_args.number_of_molecules > 1:
            file_writer.writerow(
                [f"Molecule {i}" for i in range(train_args.number_of_molecules)] + ["Predicted Bliss Score"]
            )

        for i, datapoint in tqdm(enumerate(dataset), desc="Molecules predicted:", leave=False, total=len(dataset)):
            output = m.forward(datapoint).detach().cpu().numpy()[0]
            if train_args.number_of_molecules > 1:
                file_writer.writerow([f"{mol}" for mol in datapoint.smiles_list] + [output])
            else:
                file_writer.writerow([datapoint.smiles, output])


train_model(sys.argv[1], sys.argv[2], sys.argv[3])
