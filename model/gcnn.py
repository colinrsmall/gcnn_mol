import torch
from args import ModelArgs
from data.dataset import Dataset


class GCNN(torch.nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()

        self.model_args = model_args

        #
