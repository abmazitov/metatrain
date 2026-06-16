# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import List, Optional

import torch
from metatomic.torch import System
from typing_extensions import TypedDict

from metatrain.utils.neighbor_lists import NeighborListOptions

try:
    from torchpme.lib import generate_kvectors_for_ewald, lr_wavelength_for_num_k
except ImportError:
    # torch-pme is an optional dependency, only required when long-range interactions
    # are enabled. These names are referenced by ``_capped_batched_kvectors`` below,
    # which only runs when ``num_k > 0`` (i.e. when torch-pme is installed and the
    # Ewald calculator has been constructed).
    pass


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
    num_k: int = 3000
    """Target number of half-space reciprocal-space vectors per structure for the
    batched Ewald sum (training only). When ``> 0``, the per-structure k-vector count is
    capped at this value (and half-space summation is enabled), which bounds the memory
    of batches that mix very differently sized cells -- the regime in which a fixed
    ``kspace_resolution`` lets the largest cell in the batch dictate (and inflate) the
    padded k-vector count for every structure. ``0`` keeps the original
    behavior of deriving the k-vectors from ``kspace_resolution`` alone."""


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

        self.num_k = int(hypers["num_k"])
        """Cap on the per-structure half-space k-vector count for the batched Ewald sum
        (``0`` disables the cap). See :class:`LongRangeHypers`."""

        self.kspace_resolution = float(hypers["kspace_resolution"])
        """Reciprocal-space resolution used as the (finest) floor when capping the
        k-vector count with :attr:`num_k`."""

        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(hypers["kspace_resolution"]),
            halfspace=self.num_k > 0,
        )
        """Calculator to compute the long-range electrostatic potential using the Ewald
        summation method. When :attr:`num_k` is set, it is constructed with
        ``halfspace=True`` and fed explicit, count-capped k-vectors at evaluation
        time."""

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
            # When ``num_k`` is set, generate explicit per-structure k-vectors whose
            # count is capped at ``num_k`` (reusing the shared, fixed neighbor list
            # unchanged); otherwise let the calculator derive them from its
            # ``lr_wavelength``. Capping bounds the padded k-dimension -- and hence the
            # reciprocal-space memory -- on batches with heterogeneous cell sizes.
            kvectors: Optional[torch.Tensor] = None
            if self.num_k > 0:
                kvectors = _capped_batched_kvectors(
                    cells, pbc, self.num_k, self.kspace_resolution
                )
            potential = self.ewald_calculator.forward(
                charges=charges,
                cell=cells,
                positions=positions,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                system_index=system_indices,
                periodic=pbc,
                kvectors=kvectors,
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


def _capped_batched_kvectors(
    cells: torch.Tensor,
    periodic: torch.Tensor,
    num_k: int,
    kspace_resolution: float,
) -> torch.Tensor:
    """Generate zero-padded per-structure Ewald k-vectors with a capped count.

    For each structure the reciprocal-space resolution is the *coarser* of the
    configured ``kspace_resolution`` and the resolution that would yield ``num_k``
    half-space k-vectors for that cell (:func:`lr_wavelength_for_num_k`). Taking the
    coarser of the two means small and medium cells keep the configured resolution
    exactly, while only the cells large enough to exceed ``num_k`` vectors are coarsened
    down to ``num_k``. This bounds the padded k-dimension -- and hence the memory of the
    batched reciprocal-space sum -- to roughly ``num_k`` regardless of how different the
    cell sizes in the batch are.

    The smearing is intentionally *not* adjusted per structure: it is fixed by the
    calculator and must stay consistent with the fixed real-space cutoff of the shared
    neighbor list. As a consequence ``num_k`` must be chosen large enough that the
    largest cells remain converged at that smearing, otherwise their reciprocal-space
    accuracy degrades (the smaller cells, kept at ``kspace_resolution``, are unaffected).

    Non-periodic (0D) structures get a single zero k-vector; their reciprocal-space
    contribution is masked out by the calculator.

    :param cells: tensor of shape ``(B, 3, 3)`` with the per-structure lattice vectors.
    :param periodic: bool tensor of shape ``(B, 3)`` with the per-direction periodicity.
    :param num_k: target/cap on the number of half-space k-vectors per structure.
    :param kspace_resolution: reciprocal-space resolution used as the finest floor.

    :return: tensor of shape ``(B, max_k, 3)`` of zero-padded k-vectors.
    """
    all_kvectors: List[torch.Tensor] = []
    for index in range(cells.shape[0]):
        cell = cells[index]
        if not bool(torch.any(periodic[index])):
            all_kvectors.append(
                torch.zeros((1, 3), dtype=cell.dtype, device=cell.device)
            )
            continue
        lr_num_k = lr_wavelength_for_num_k(cell, num_k)
        lr_eff = lr_num_k if lr_num_k > kspace_resolution else kspace_resolution
        ns = torch.ceil(torch.linalg.norm(cell, dim=-1) / lr_eff).long()
        all_kvectors.append(
            generate_kvectors_for_ewald(cell=cell, ns=ns, halfspace=True)
        )
    return torch.nn.utils.rnn.pad_sequence(all_kvectors, batch_first=True)


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
