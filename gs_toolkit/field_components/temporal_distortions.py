"""Space distortions which occur as a function of time."""

import abc
from enum import Enum
from typing import Any, Dict, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from gs_toolkit.field_components.encodings import Encoding, NeRFEncoding
from gs_toolkit.field_components.mlp import MLP


class TemporalDistortion(nn.Module):
    """Apply spatial distortions as a function of time"""

    @abc.abstractmethod
    def forward(
        self, positions: Float[Tensor, "*bs 3"], times: Float[Tensor, "*bs 1"]
    ) -> Float[Tensor, "*bs 3"]:
        """
        Args:
            positions: Samples to translate as a function of time
            times: times for each sample

        Returns:
            Translated positions.
        """


class TemporalDistortionKind(Enum):
    """Possible temporal distortion names"""

    DNERF = "dnerf"

    def to_temporal_distortion(self, config: Dict[str, Any]) -> TemporalDistortion:
        """Converts this kind to a temporal distortion"""
        if self == TemporalDistortionKind.DNERF:
            return DNeRFDistortion(**config)
        raise NotImplementedError(f"Unknown temporal distortion kind {self}")


class DNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=10,
            min_freq_exp=0.0,
            max_freq_exp=8.0,
            include_input=True,
        ),
        temporal_encoding: Encoding = NeRFEncoding(
            in_dim=1,
            num_frequencies=10,
            min_freq_exp=0.0,
            max_freq_exp=8.0,
            include_input=True,
        ),
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.mlp_deform = MLP(
            in_dim=self.position_encoding.get_out_dim()
            + self.temporal_encoding.get_out_dim(),
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(
        self, positions: Float[Tensor, "*bs 3"], times: Float[Tensor, "*bs 1"]
    ) -> Float[Tensor, "*bs 3"]:
        p = self.position_encoding(positions)
        t = self.temporal_encoding(times)
        return self.mlp_deform(torch.cat([p, t], dim=-1))
