from utils.args import TrainArgs
from data.dataset import load_dataset
from model import gcnn
from torch import nn
from tqdm.auto import tqdm
from utils import metrics, utils

import torch
import torch.optim as optim

import wandb


def train_model(train_args: TrainArgs):
    # Initialize wandb logging
    if train_args.wandb_logging:
        wandb.init(project="gcnn_mol")
        wandb.config = {
            "epochs": train_args.epochs,
            "depth": train_args.depth,
            "hidden_size": train_args.hidden_size,
            "activation_function": train_args.activation_function,
            "readout_num_hidden_layers": train_args.readout_num_hidden_layers,
            "readout_hidden_size": train_args.readout_hidden_size,
            "bias": train_args.bias,
            "node_level_dropout": train_args.node_level_dropout,
            "readout_dropout": train_args.readout_dropout,
            "dropout_prob": train_args.dropout_probability,
            "aggregation_method": train_args.aggregation_method,
            "shared+node_level_nns": train_args.shared_node_level_nns,
            "mol_features_only": train_args.mol_features_only,
            "explicit_hydrogens": train_args.explicit_hydrogens,
            "atom_descriptors": train_args.atom_descriptors,
            "molecule_descriptors": train_args.molecule_descriptors,
        }

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data according to train_args settings
    dataset = load_dataset(train_args)

    # Scale dataset features
    feature_scalers = dataset.fit_scalers_to_features()
    dataset.normalize_features(feature_scalers)

    # Move dateset to detected device
    dataset.to(device)

    # Split data in to training and test set
    train_set, test_set = dataset.train_test_split(0.2)

    # Build model to use
    if train_args.mol_features_only:
        m = gcnn.FCNNOnly(
            train_args,
            dataset.mol_features_vector_length,
            train_args.number_of_molecules,
        ).to(device)
    else:
        m = gcnn.GCNN(
            train_args,
            dataset.atom_features_vector_length,
            train_args.number_of_molecules,
            dataset.mol_features_vector_length,
            device,
        ).to(device)

    # Initiate model optimizer and loss function
    optimizer = optim.SGD(
        m.parameters(),
        lr=train_args.sgd_lr,
        momentum=train_args.sgd_momentum,
        nesterov=train_args.sgd_nesterov,
        dampening=train_args.sgd_dampening,
        weight_decay=train_args.sgd_weight_decay,
    )

    match train_args.loss_function:
        case "mae":
            loss_function = nn.L1Loss()
        case "mse":
            loss_function = nn.MSELoss()
        case _:
            raise ValueError(f"{train_args.loss_function} not implemented.")

    # Initiate dictionaries to save test and training metrics in
    training_metrics = {metric: 0 for metric in train_args.metrics}
    test_metrics = {metric: 0 for metric in train_args.metrics}
    model_save_metric = 0

    # Run model training
    if train_args.wandb_logging:
        wandb.watch(m, loss_function, "all")

    pbar = tqdm(range(train_args.epochs))
    torch.autograd.set_detect_anomaly(True)
    for epoch in pbar:
        training_outputs = []
        training_targets = [dp.target for dp in train_set]
        test_targets = [dp.target for dp in test_set]
        running_loss = 0

        # Train model on training set
        for i, datapoint in tqdm(enumerate(train_set), desc="Train set:", leave=False, total=len(train_set)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass of model
            output = m.forward(datapoint)
            training_outputs.append(output.detach().numpy())

            # Calculate loss, backprop, and update optimizer
            # Check if output and target will produce 0, fudge output if so
            if output == torch.Tensor([datapoint.target]):
                output += 1e-4
            loss = loss_function(output, torch.Tensor([datapoint.target]))
            running_loss = loss.item()
            loss.backward()
            optimizer.step()

        # Calculate training set metrics
        for metric in train_args.metrics:
            metric_func = metrics.all_metrics()[metric]
            metric_val = metric_func(training_outputs, training_targets)
            training_metrics[metric] = metric_val

        test_outputs = []

        # Evaluate model on test set
        for datapoint in tqdm(test_set, desc="Test set:", leave=False, total=len(test_set)):
            # Forward pass of model
            output = m.forward(datapoint)
            test_outputs.append(output.detach().numpy())

        # Calculate test set metrics
        for metric in train_args.metrics:
            metric_func = metrics.all_metrics()[metric]
            metric_val = metric_func(test_outputs, test_targets)
            test_metrics[metric] = metric_val

        # Save model if better performing in model_save_metric
        if test_metrics[train_args.model_save_metric] > model_save_metric:
            if train_args.model_save_path is not None:
                model_save_metric = test_metrics[train_args.model_save_metric]
                print(
                    f"Saving model on epoch {epoch} with new best {train_args.model_save_metric} of {model_save_metric}."
                )
                utils.save_checkpoint(m, feature_scalers, train_args)

        # Print model metrics
        metric_description = "; ".join([f"{metric}: {test_metrics[metric]:.4f}" for metric in train_args.metrics])
        pbar.set_description(metric_description)

        # Log with wandb is using
        if train_args.wandb_logging:
            log_dict = (
                {f"training_{metric}": value for metric, value in training_metrics.items()}
                | {f"test_{metric}": value for metric, value in test_metrics.items()}
                | {"loss": running_loss}
            )
            wandb.log(log_dict)
