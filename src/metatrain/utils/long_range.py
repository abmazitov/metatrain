# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import List

import torch
from metatomic.torch import System
from typing_extensions import TypedDict

from metatrain.utils.neighbor_lists import NeighborListOptions


class LongRangeHypers(TypedDict):
    """In some systems and datasets, enabling long-range Coulomb interactions
    might be beneficial for the accuracy of the model and/or
    its physical correctness."""

    enable: bool = False
    """Toggle for enabling long-range interactions"""
    use_ewald: bool = False
    """Use Ewald summation. If False, P3M is used"""
    smearing: float = 1.4
    """Smearing width in Fourier space"""
    kspace_resolution: float = 1.33
    """Resolution of the reciprocal space grid"""
    interpolation_nodes: int = 5
    """Number of grid points for interpolation (for P3M/PME only)"""


class LongRangeFeaturizer(torch.nn.Module):
    """A class to compute long-range features starting from short-range features.

    :param hypers: Dictionary containing the hyperparameters for the long-range
        featurizer.
    :param neighbor_list_options: A :py:class:`NeighborListOptions` object containing
        the neighbor list information for the short-range model.
    :param feature_dim: The dimension of the short-range features (which also
        corresponds to the number of long-range features that will be returned).
    :param output_dim: The dimension of the long-range features that will be returned.

    """

    def __init__(
        self,
        hypers: LongRangeHypers,
        neighbor_list_options: NeighborListOptions,
        feature_dim: int,
        output_dim: int,
    ) -> None:
        super(LongRangeFeaturizer, self).__init__()

        try:
            from torchpme import (
                CoulombPotential,
                EwaldCalculator,
                P3MCalculator,
            )
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install 'torch-pme>=0.3.2'`."
            )

        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(hypers["kspace_resolution"]),
        )
        """Calculator to compute the long-range electrostatic potential using the Ewald
        summation method."""

        self.p3m_calculator = P3MCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=neighbor_list_options.full_list,
            mesh_spacing=float(hypers["kspace_resolution"]),
        )
        """Calculator to compute the long-range electrostatic potential using the P3M
        method."""

        self.use_ewald = hypers["use_ewald"]
        """If ``True``, use the Ewald summation method instead of the P3M method for
        periodic systems during training."""

        self.charges_map = torch.nn.Linear(feature_dim, feature_dim)
        """Map the short-range features to atomic charges."""

        self.out_projection = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(feature_dim, output_dim),
        )

    def forward(
        self,
        systems: List[System],
        node_features: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        system_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the long-range features for a list of systems.

        :param systems: A list of :py:class:`System` objects for which to compute the
            long-range features. Each system must contain a neighbor list consistent
            with the neighbor list options used to create the class.
        :param node_features: A tensor of short-range node features for the systems.
        :param centers: A tensor of center atom indices for the neighbor list edges.
        :param neighbors: A tensor of neighbor atom indices for the neighbor list edges.
        :param system_indices: A tensor of the system index for each atom in the batch
        :param neighbor_distances: A tensor of neighbor distances for the systems,
            which must be consistent with the neighbor list options used to create the
            class.
        :return: A tensor of long-range features for the systems.
        """
        charges = self.charges_map(node_features)
        neighbor_indices = torch.stack([centers, neighbors], dim=-1)

        positions_list: List[torch.Tensor] = []
        cells_list: List[torch.Tensor] = []
        pbc_list: List[torch.Tensor] = []
        for system in systems:
            if system.pbc.sum() == 1:
                raise NotImplementedError(
                    "Long-range featurizer does not support 1D systems."
                )
            positions_list.append(system.positions)
            n_periodic = system.pbc.sum().item()
            cell = system.cell
            if n_periodic == 2:
                cell = fill_2d_vacuum(cell, system.pbc, system.positions)
            cells_list.append(cell)
            pbc_list.append(system.pbc)

        positions = torch.concatenate(positions_list)
        pbc = torch.stack(pbc_list)
        cells = torch.stack(cells_list)

        if self.use_ewald and self.training:
            potential = self.ewald_calculator.forward(
                charges=charges,
                cell=cells,
                positions=positions,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                system_index=system_indices,
                periodic=pbc,
            )
        else:
            potential = self.p3m_calculator.forward(
                charges=charges,
                cell=cells,
                positions=positions,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                system_index=system_indices,
                periodic=pbc,
            )

        long_range_features = self.out_projection(potential)

        return long_range_features


class DummyLongRangeFeaturizer(torch.nn.Module):
    # a dummy class for torchscript
    def __init__(self) -> None:
        super().__init__()
        self.use_ewald = True

    def forward(
        self,
        systems: List[System],
        node_features: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        system_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0)


def fill_2d_vacuum(
    cell: torch.Tensor,
    periodic: torch.Tensor,
    positions: torch.Tensor,
    gap_factor: float = 1.5,
) -> torch.Tensor:
    """
    Synthesize the non-periodic lattice vector of a 2D slab cell.

    Some interfaces (notably :class:`metatomic.torch.System`) require the lattice
    vector along a non-periodic direction to be zero, which makes the cell singular.
    The Ewald slab calculation needs a non-singular cell, so this fills the missing
    vector with ``(thickness + gap_factor * L_max)`` along the plane normal -- the same
    effective height that :func:`shrink_2d_cell` would shrink a large vacuum down to,
    so the residual interaction between periodic images is negligible.

    For structures that are not 2D-periodic the cell is returned unchanged.

    :param cell: torch.tensor of shape ``(3, 3)`` with the lattice vectors as rows; the
        non-periodic row is expected to be (close to) zero.
    :param periodic: torch.tensor of shape ``(3,)`` and dtype bool.
    :param positions: torch.tensor of shape ``(N, 3)`` with the atomic positions.
    :param gap_factor: multiple of the longer in-plane lattice vector to use as the
        vacuum gap.

    :return: the cell of shape ``(3, 3)`` with the non-periodic vector filled in.
    """
    if int(periodic.to(torch.int64).sum()) != 2:
        return cell

    axis = int(torch.argmax((~periodic).to(torch.int64)))
    v1, v2 = cell[(axis + 1) % 3], cell[(axis + 2) % 3]
    normal = torch.linalg.cross(v1, v2)
    normal = normal / torch.linalg.norm(normal).clamp(min=1e-15)

    projection = positions @ normal
    thickness = projection.max() - projection.min()
    length_max = torch.maximum(torch.linalg.norm(v1), torch.linalg.norm(v2))
    height = thickness + gap_factor * length_max

    cell = cell.clone()
    cell[axis] = height * normal
    return cell
