import torch.nn as nn
from typing import Optional
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ClassicalConvNN(nn.Module):
    input_channel_number: int = 1
    kernel_size: int = 2
    padding: int = 1
    out_features: int = 2  # should be only two because QCNN has also only 2

    def __init__(
            self,
            sequence_length: int,
            num_layers: Optional[int] = None,
            num_features: Optional[int] = None
    ):
        """
        Constructs a classical convolutional neural network (CNN) that reduces
        the number of features by half in each pooling layer.

        Source:
            QCNN/CNN.py

        Parameters:
            sequence_length (int): The length of the input sequences.
            num_layers (Optional[int]): The number of convolutional and pooling
            layers. If not specified, it is calculated based on the
            sequence_length.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.num_layers = num_layers or 2
        self.num_features = num_features or 2

        logger.debug(
            f"Initializing ClassicalCNN with {self.num_layers} layers"
            f" and {self.sequence_length} sequence length."
        )

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.final_layer_size = int(self.sequence_length / (
                self.num_layers * self.kernel_size))

        for layer in range(self.num_layers):
            if layer == 0:
                conv = nn.Conv1d(self.input_channel_number, self.num_features,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)
            else:
                conv = nn.Conv1d(self.num_features, self.num_features,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)
            self.convs.append(conv)
            pool = nn.MaxPool1d(kernel_size=self.kernel_size)
            self.pools.append(pool)
            logger.debug(f"Layer {layer} initialized: Conv1d and MaxPool1d.")

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): The input tensor of shape (batch_size,
        input_channels, seq_length).

        Returns:
        torch.Tensor: The output tensor after passing through the CNN layers.
        """
        for layer in range(self.num_layers):
            x = self.convs[layer](x)
            x = nn.ReLU()(x)
            x = self.pools[layer](x)
        x = nn.Flatten()(x)
        x = nn.Linear(self.num_features * self.final_layer_size,
                      self.out_features)(x)
        return x

    @property
    def num_parameters(self):
        pass

    @num_parameters.getter
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def ordered_dict(self):
        pass

    @ordered_dict.getter
    def ordered_dict(self) -> OrderedDict:
        dictionary = OrderedDict()
        for layer in range(self.num_layers):
            dictionary[f"conv{layer}"] = self.convs[layer]
            dictionary[f"relu{layer}"] = nn.ReLU()
            dictionary[f"pool{layer}"] = self.pools[layer]
        dictionary["flatten"] = nn.Flatten()
        dictionary["linear"] = nn.Linear(
            self.num_features * self.final_layer_size,
            self.out_features)
        return dictionary

    @property
    def ccnn(self):
        """classical convolutional neural network"""
        pass

    @ccnn.getter
    def ccnn(self) -> nn.Sequential:
        return nn.Sequential(self.ordered_dict)
