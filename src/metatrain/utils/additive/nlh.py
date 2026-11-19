import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    unit_conversion_factor,
)

from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from ..abc import ModelInterface
from ..data import DatasetInfo, TargetInfo
from ..sum_over_atoms import sum_over_atoms


# e^2 / (4 pi eps0), in eV * Angstrom
_COULOMB_CONSTANT = 14.399645478425668

_COEFFICIENTS_FILE = Path(__file__).parent / "nlh_coefficients.dat"

# The NLH paper tabulates all pairs with 1 <= Z <= 92
_MAX_TABULATED_Z = 92

# Fraction of the switching radius below which the NLH potential is used unmodified
# (fitted together with the radii below, see there)
_SWITCH_ONSET = 0.673

# Per-element repulsion radii (Angstrom), from H (Z=1) to No (Z=102). The NLH
# potential between atoms i and j is used unmodified below ``r_on = _SWITCH_ONSET *
# r_c``, and multiplied by a C2 "smootherstep" switching function going from 1 at
# ``r_on`` to 0 at ``r_c = R_i + R_j``.
#
# The radii and the onset were fitted once, offline, on the training split of the
# r2SCAN MAD 2.0 dataset (version 2.0rc5), with the following procedure:
#
# 1. Data: all 842,134 structures (23.6M atoms) of the training split, in 28
#    subsets; only the "monomers" subset (single atoms) is excluded. The "dimers"
#    subset contains dimer curves of every pair of elements, reaching into the
#    repulsive wall.
# 2. Parametrization: R_i = R_cov,i * (0.4 + 1.1 * sigmoid(theta_i)), i.e. each
#    radius is constrained to [0.4, 1.5] x R_cov,i. R_cov are the covalent radii
#    of Cordero et al. (as in ``ase.data.covalent_radii``), completed for Bk-No
#    (which ASE lacks) with the single-bond radii of Pyykko and Atsumi: 1.70 (Bk,
#    interpolated), 1.72, 1.66, 1.66, 1.68 and 1.68 Angstrom. The onset is
#    _SWITCH_ONSET = sigmoid(tau). This gives 103 parameters in total.
# 3. Loss: for every subset g, the relative residual force
#    L_g = sum_atoms |F_DFT - F_NLH|^2 / sum_atoms |F_DFT|^2, averaged over all
#    subsets with equal weight, plus a penalty 1e-3 * sum_i (theta_i - theta_0)^2
#    pulling the radii towards 0.8 x R_cov,i. Giving every subset the same weight
#    balances the two goals of the potential: near-equilibrium subsets (small
#    forces) penalize any leaking of the repulsion into bonded configurations, and
#    so push the radii down, while compressed, rattled and dimer subsets reward
#    removing the steep repulsive wall, and push them up. Energies are not part of
#    the loss.
# 4. Optimization, in two stages. First, a preliminary fit starting from
#    R_i = 0.8 x R_cov,i and _SWITCH_ONSET = 0.5, on a ~1% subsample of the
#    training split (Adam, learning rate 0.05, 300 full-batch steps). Then,
#    starting from that fit, the final fit on the full training split (Adam,
#    learning rate 0.01, cosine-decayed to 1e-3, 400 steps), where each step uses
#    256 random structures of every subset (all of them for smaller subsets), with
#    weights n_g / 256 making the batch loss an unbiased estimate of the full loss,
#    and neighbor lists computed from scratch for every batch. The radii were
#    rounded to 1e-3 Angstrom.
#
# On the 169,758 held-out structures of the validation and test splits, the
# relative RMS of the residual forces is 0.94 on average over near-equilibrium
# subsets (3.5 for the ZBL potential switched off at the covalent radii, which
# leaks into bonded configurations), and 0.59 over compressed and rattled subsets
# (1.15 for ZBL). The potential is not forced to vanish at equilibrium bond
# lengths: over all of MAD 2.0rc5, it acts on 2.2% of the atoms of relaxed
# structures (mc3d, mc3d_ext and shiftml subsets), and exceeds 0.5 eV/A for 0.13%
# of them, mostly for short metal-metal and actinyl bonds of d and f elements.
# Such small and smooth contributions are left to the machine learning model. The
# radii should be refitted if the potential is used as a baseline for very
# different reference data.
# fmt: off
_REPULSION_RADII = [
    0.410, 0.230, 1.045, 0.779, 0.725, 0.618, 0.556, 0.547,
    0.570, 0.489, 1.300, 1.220, 1.125, 1.059, 0.953, 0.898,
    0.885, 0.928, 1.656, 1.473, 1.391, 1.259, 1.050, 1.055,
    1.042, 1.087, 0.990, 1.074, 1.196, 1.085, 1.105, 1.160,
    1.120, 1.104, 1.078, 1.038, 1.788, 1.649, 1.539, 1.429,
    1.238, 1.166, 1.055, 1.094, 1.118, 1.237, 1.358, 1.298,
    1.318, 1.303, 1.307, 1.334, 1.268, 1.257, 1.952, 1.776,
    1.507, 1.456, 1.428, 1.458, 1.633, 1.615, 1.608, 1.575,
    1.552, 1.530, 1.568, 1.556, 1.566, 1.588, 1.486, 1.411,
    1.300, 1.192, 1.181, 1.100, 1.141, 1.210, 1.254, 1.275,
    1.344, 1.365, 1.414, 1.290, 1.332, 1.415, 2.109, 1.779,
    1.678, 1.419, 1.529, 1.297, 1.368, 1.350, 1.406, 1.376,
    1.396, 1.401, 1.345, 1.364, 1.379, 1.395,
]
# fmt: on
_MAX_Z = len(_REPULSION_RADII)


@lru_cache(maxsize=1)
def _tabulated_coefficients() -> Dict[Tuple[int, int], Tuple[List[float], List[float]]]:
    """Read the NLH coefficient table shipped with metatrain.

    :return: A dictionary mapping ``(Z1, Z2)`` with ``Z1 <= Z2`` to the ``a`` and
        ``b`` coefficients of the screening function.
    """
    table: Dict[Tuple[int, int], Tuple[List[float], List[float]]] = {}
    with open(_COEFFICIENTS_FILE, "r") as fd:
        for line in fd:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            z1, z2 = sorted((int(fields[0]), int(fields[1])))
            values = [float(x) for x in fields[2:8]]
            table[(z1, z2)] = (values[0::2], values[1::2])
    return table


def _screening_length(z1: int, z2: int) -> float:
    return 0.46850 / (z1**0.23 + z2**0.23)


def nlh_coefficients(z1: int, z2: int) -> Tuple[List[float], List[float]]:
    """
    Coefficients of the NLH screening function for a pair of elements.

    The NLH potential reads :math:`V(r) = \\frac{Z_1 Z_2 e^2}{4 \\pi \\epsilon_0 r}
    \\sum_{k=1}^3 a_k e^{-b_k r}`, with :math:`r` in Å and :math:`b_k` in Å⁻¹.

    Pairs with :math:`Z \\le 92` use the published coefficients. Elements beyond
    uranium are not covered by the NLH paper: for those, the coefficients of the pair
    where every element heavier than uranium is replaced by uranium are used, and the
    exponents :math:`b_k` are rescaled by the ratio of the ZBL screening lengths
    :math:`a = 0.4685 / (Z_1^{0.23} + Z_2^{0.23})` of the two pairs. The Coulomb
    prefactor always uses the true atomic numbers.

    :param z1: Atomic number of the first element.
    :param z2: Atomic number of the second element.
    :return: The ``a`` and ``b`` coefficients (three of each).
    """
    for z in (z1, z2):
        if z < 1 or z > _MAX_Z:
            raise ValueError(
                f"the NLH potential is only available for elements with 1 <= Z <= "
                f"{_MAX_Z}, got Z={z}"
            )

    proxy_1, proxy_2 = sorted((min(z1, _MAX_TABULATED_Z), min(z2, _MAX_TABULATED_Z)))
    a, b = _tabulated_coefficients()[(proxy_1, proxy_2)]
    scale = _screening_length(proxy_1, proxy_2) / _screening_length(z1, z2)
    return list(a), [b_k * scale for b_k in b]


def nlh_switching_radius(z1: int, z2: int) -> float:
    """
    Distance (in Å) at which the NLH potential of a pair is switched off.

    This is the sum of the repulsion radii of the two elements, fitted to the r2SCAN
    MAD 2.0 dataset.

    :param z1: Atomic number of the first element.
    :param z2: Atomic number of the second element.
    :return: The switching radius of the pair.
    """
    for z in (z1, z2):
        if z < 1 or z > _MAX_Z:
            raise ValueError(
                f"the NLH potential is only available for elements with 1 <= Z <= "
                f"{_MAX_Z}, got Z={z}"
            )
    return _REPULSION_RADII[z1 - 1] + _REPULSION_RADII[z2 - 1]


class NLH(ModelInterface):
    """
    Short-range repulsive potential of Nordlund, Lehtola and Hobler (NLH).

    The NLH potential is a pair-specific screened Coulomb repulsion, fitted to
    all-electron DFT dimer calculations for every pair of elements with
    :math:`Z \\le 92` (`Phys. Rev. A 111, 032818 (2025)
    <https://doi.org/10.1103/PhysRevA.111.032818>`_). Compared to the universal ZBL
    potential, it is several times more accurate in the repulsive regime, especially
    for transition metals and lanthanides, where ZBL is far too repulsive at the
    distances relevant for atomistic simulations. Elements up to nobelium
    (:math:`Z = 102`) are supported by extrapolation, see
    :py:func:`nlh_coefficients`.

    The potential is used unmodified below :math:`0.673 \\, r_c` and multiplied by a
    smooth (:math:`C^2`) switching function going to zero at :math:`r_c = R_i + R_j`,
    the sum of per-element radii (see :py:func:`nlh_switching_radius`). The radii are
    fitted to the r2SCAN MAD 2.0 dataset, so that the potential removes most of the
    steep short-range repulsion from the data while barely affecting bonded
    configurations, which are left to the machine learning model.

    :param hypers: A dictionary of model hyperparameters. This parameter is ignored and
        is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    species_to_index: torch.Tensor
    coefficients_a: torch.Tensor
    coefficients_b: torch.Tensor
    switching_radii: torch.Tensor

    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "K. Nordlund, S. Lehtola, G. Hobler, Phys. Rev. A 111, 032818 (2025)",
            ]
        }
    )

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)

        # `hypers` should be an empty dictionary
        if not (isinstance(hypers, dict) and len(hypers) == 0):
            raise ValueError(
                f"{self.__class__.__name__} hypers takes an empty dictionary. "
                f"Got: {hypers}."
            )

        length_unit = dataset_info.length_unit
        if length_unit == "" or unit_conversion_factor(length_unit, "angstrom") != 1.0:
            raise ValueError(
                "NLH only supports angstrom units, but a "
                f"'{length_unit}' unit was provided."
            )

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"NLH model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.dataset_info = dataset_info
        self.outputs: Dict[str, TargetInfo] = dict(dataset_info.targets)
        self.switch_onset = _SWITCH_ONSET
        self.coulomb_constant = _COULOMB_CONSTANT

        self.register_buffer("species_to_index", torch.empty(0, dtype=torch.int))
        self.register_buffer("coefficients_a", torch.empty(0, dtype=torch.float64))
        self.register_buffer("coefficients_b", torch.empty(0, dtype=torch.float64))
        self.register_buffer("switching_radii", torch.empty(0, dtype=torch.float64))
        self.cutoff_radius = 0.0
        self._set_atomic_types(dataset_info.atomic_types)

    def _set_atomic_types(self, atomic_types: List[int]) -> None:
        self.atomic_types = sorted(atomic_types)
        n_types = len(self.atomic_types)
        device = self.species_to_index.device

        species_to_index = torch.full(
            (max(self.atomic_types) + 1,), -1, dtype=torch.int
        )
        coefficients_a = torch.zeros((n_types, n_types, 3), dtype=torch.float64)
        coefficients_b = torch.zeros((n_types, n_types, 3), dtype=torch.float64)
        switching_radii = torch.zeros((n_types, n_types), dtype=torch.float64)

        for i, z_i in enumerate(self.atomic_types):
            species_to_index[z_i] = i
            for j, z_j in enumerate(self.atomic_types):
                a, b = nlh_coefficients(z_i, z_j)
                coefficients_a[i, j] = torch.tensor(a, dtype=torch.float64)
                coefficients_b[i, j] = torch.tensor(b, dtype=torch.float64)
                switching_radii[i, j] = nlh_switching_radius(z_i, z_j)

        if any(z > _MAX_TABULATED_Z for z in self.atomic_types):
            logging.info(
                "NLH coefficients for elements beyond uranium (Z > 92) are "
                "extrapolated from the uranium ones."
            )

        self.species_to_index = species_to_index.to(device)
        self.coefficients_a = coefficients_a.to(device)
        self.coefficients_b = coefficients_b.to(device)
        self.switching_radii = switching_radii.to(device)
        self.cutoff_radius = float(torch.max(switching_radii))

    def restart(
        self, dataset_info: DatasetInfo, model_hypers: Mapping[str, Any] | None = None
    ) -> "NLH":
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"NLH model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.dataset_info = self.dataset_info.union(dataset_info)
        self.outputs.update(dataset_info.targets)
        if set(self.dataset_info.atomic_types) != set(self.atomic_types):
            self._set_atomic_types(self.dataset_info.atomic_types)
        return self

    def remove_output(self, target_name: str) -> None:
        """
        Remove a previously registered output target.

        :param target_name: Name of the target to remove.
        """
        self.outputs.pop(target_name, None)
        self.dataset_info.targets.pop(target_name, None)

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return self.outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the energies of a system solely based on the NLH repulsive
        potential.

        :param systems: List of systems to calculate the NLH energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :return: A dictionary with the computed predictions for each system.
        """
        nl_options = self.requested_neighbor_lists()[0]
        neighbor_lists: List[TensorBlock] = []
        for system in systems:
            neighbor_lists.append(system.get_neighbor_list(nl_options))

        zi = torch.concatenate(
            [
                system.types[nl.samples.column("first_atom")]
                for nl, system in zip(neighbor_lists, systems, strict=True)
            ]
        )
        zj = torch.concatenate(
            [
                system.types[nl.samples.column("second_atom")]
                for nl, system in zip(neighbor_lists, systems, strict=True)
            ]
        )
        rij = torch.concatenate(
            [torch.sqrt(torch.sum(nl.values**2, dim=(1, 2))) for nl in neighbor_lists]
        )

        # the neighbor list is full, so each pair is counted twice
        e_pairs = 0.5 * self.pair_energy(zi, zj, rij)

        indices_for_sum_list = []
        n_atoms = 0
        for system, nl in zip(systems, neighbor_lists, strict=True):
            indices_for_sum_list.append(nl.samples.column("first_atom") + n_atoms)
            n_atoms += system.positions.shape[0]

        e_nodes = torch.zeros(n_atoms, dtype=e_pairs.dtype, device=e_pairs.device)
        e_nodes.index_add_(0, torch.cat(indices_for_sum_list), e_pairs)

        device = systems[0].positions.device

        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            sample_values: List[List[int]] = []
            for i_system, system in enumerate(systems):
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]

            block = TensorBlock(
                values=e_nodes.reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor(sample_values, device=device),
                    assume_unique=True,
                ),
                components=[],
                properties=Labels(
                    names=["energy"], values=torch.tensor([[0]], device=device)
                ),
            )

            targets_out[target_key] = TensorMap(
                keys=Labels(names=["_"], values=torch.tensor([[0]], device=device)),
                blocks=[block],
            )

            if selected_atoms is not None:
                targets_out[target_key] = mts.slice(
                    targets_out[target_key], "samples", selected_atoms
                )

            if target.sample_kind == "system":
                targets_out[target_key] = sum_over_atoms(targets_out[target_key])

        return targets_out

    def pair_energy(
        self, zi: torch.Tensor, zj: torch.Tensor, rij: torch.Tensor
    ) -> torch.Tensor:
        """
        Switched NLH energy of isolated atom pairs.

        :param zi: Atomic numbers of the first atoms.
        :param zj: Atomic numbers of the second atoms.
        :param rij: Distances between the atoms, in Å.
        :return: The energy (in eV) of each pair.
        """
        ti = self.species_to_index[zi]
        tj = self.species_to_index[zj]
        a = self.coefficients_a[ti, tj].to(rij.dtype)
        b = self.coefficients_b[ti, tj].to(rij.dtype)
        r_cut = self.switching_radii[ti, tj].to(rij.dtype)
        r_on = self.switch_onset * r_cut

        phi = torch.sum(a * torch.exp(-b * rij.unsqueeze(-1)), dim=-1)
        energy = self.coulomb_constant * (zi * zj).to(rij.dtype) / rij * phi

        # C2 "smootherstep" switching function, 1 below r_on and 0 above r_cut
        x = torch.clamp((rij - r_on) / (r_cut - r_on), 0.0, 1.0)
        switch = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

        return energy * switch

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
                strict=True,
            )
        ]

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a :py:class:`TargetInfo` object is compatible with the NLH model.

        :param target_name: The name of the target to be checked.
        :param target_info: The :py:class:`TargetInfo` object to be checked.
        :return: True if the target is compatible with the NLH model, False otherwise.
        """
        if target_info.quantity != "energy":
            logging.debug(
                f"NLH model does not support target {target_name} since it is "
                "not an energy."
            )
            return False
        if not target_info.is_scalar:
            logging.debug(
                f"NLH model does not support target {target_name} since it is "
                "not a scalar."
            )
            return False
        if len(target_info.layout.block(0).properties) > 1:
            logging.debug(
                f"NLH model does not support target {target_name} since it has "
                "more than one property."
            )
            return False
        if target_info.unit != "eV":
            logging.debug(
                f"NLH model does not support target {target_name} since it is "
                "not in eV."
            )
            return False
        return True

    def get_checkpoint(
        self, best_model_state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        checkpoint = {
            "architecture_name": "nlh",
            "model_ckpt_version": self.__checkpoint_version__,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": self.state_dict(),
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "NLH":
        """
        Create a model from a checkpoint (i.e. state dictionary).

        :param checkpoint: Checkpoint's state dictionary.
        :param context: Context in which to load the model. Not used.

        :return: An instance of the model.
        """
        model = cls(
            hypers=checkpoint["model_data"]["hypers"],
            dataset_info=checkpoint["model_data"]["dataset_info"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict, version: Optional[int] = None
    ) -> dict:
        """
        Upgrade the checkpoint to the current version of the model.

        :param checkpoint: Checkpoint's state dictionary.
        :param version: Version to which the checkpoint should be upgraded. If ``None``,
            the checkpoint will be upgraded to the current version of the model.

        :return: The upgraded checkpoint.
        """
        if version is None:
            version = cls.__checkpoint_version__
        if checkpoint["model_ckpt_version"] != version:
            raise NotImplementedError(
                "NLH model does not support checkpoint upgrading. "
                "There is only one checkpoint version."
            )
        return checkpoint

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        """
        Turn this model into an :py:class:`metatomic.torch.AtomisticModel`,
        containing the model itself, its capabilities and its metadata.

        :param metadata: Additional metadata to add to the model, as specified
            by the user.
        :return: An instance of :py:class:`metatomic.torch.AtomisticModel`.
        """
        capabilities = ModelCapabilities(
            outputs={
                name: info.to_model_output(sample_kind="atom")
                for name, info in self.outputs.items()
            },
            atomic_types=self.atomic_types,
            interaction_range=self.cutoff_radius,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(torch.float32),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)
