from torch import nn
import torch
from sklearn.preprocessing import StandardScaler
from . import args
import os
import subprocess

MODEL_SAVE_PREFIX = "gcnn_mol_trained_"


def save_checkpoint(
    model: nn.Module,
    atom_features_scalers: list[StandardScaler],
    train_args: args.TrainArgs,
) -> None:
    """
    Save a gcnn_mol model, atom feature scalers, and training arguments to a given path as specified in train_args.
    :param model: The gcnn model to save.
    :param atom_features_scalers: The fitted atom feature scalers to save.
    :param train_args: The training arguments the model was built and trained with.
    """
    # Create directories to save path if they do not exist
    save_path = os.path.join(train_args.model_save_path, train_args.model_save_name)
    os.makedirs(save_path)

    # Taken from: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()

    checkpoint_state = {
        "train_args": train_args,
        "state_dict": model.state_dict(),
        "atom_features_scalers": atom_features_scalers,
        "git_hash": git_hash,  # Saved for debugging, later use if needed
    }
    torch.save(checkpoint_state, save_path)


def load_checkpoint(load_path: str) -> (dict, list[StandardScaler], args.TrainArgs):
    """
    Load a gcnn_mol model, atom feature scalers, and training arguments from a given path.
    :param load_path: The path from which to load the model.
    :return: The loaded model's state dict (which contains the model's weights), atom feature scalers, and training
             arguments that were used to build the model.
    """
    load_path = os.path.join(load_path)

    if not os.path.exists(load_path):
        raise ValueError(f"Model at {load_path} does not exist.")

    state = torch.load(load_path)
    train_args = state["train_args"]
    state_dict = state["state_dict"]
    atom_features_scalers = state["atom_features_scalers"]

    return state_dict, atom_features_scalers, train_args


class NoamOpt:
    """
    Optim wrapper that implements rate.
    From https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch and
    https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    """

    def __init__(self, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
