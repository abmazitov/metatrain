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

# Elements supported by the NLH potential, from H to No
_MAX_Z = 102

# Shortest distance between two atoms of every pair of elements in the training data
# of PET-MAD 2.0, see :py:func:`nlh_min_distance`
_MIN_DISTANCES_FILE = Path(__file__).parent / "nlh_min_distances.dat"

# The NLH potential of a pair vanishes above ``r_c = r_min - _CUTOFF_MARGIN`` (but at
# least ``r_min / 2``), where ``r_min`` is the shortest distance between the two
# elements in the training data, and is switched on over ``_SWITCH_WIDTH`` below
# ``r_c``, see :py:func:`nlh_switching_radii`.
_CUTOFF_MARGIN = 0.4
_SWITCH_WIDTH = 0.4


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
    _check_atomic_numbers(z1, z2)

    proxy_1, proxy_2 = sorted((min(z1, _MAX_TABULATED_Z), min(z2, _MAX_TABULATED_Z)))
    a, b = _tabulated_coefficients()[(proxy_1, proxy_2)]
    scale = _screening_length(proxy_1, proxy_2) / _screening_length(z1, z2)
    return list(a), [b_k * scale for b_k in b]


@lru_cache(maxsize=1)
def _tabulated_min_distances() -> Dict[Tuple[int, int], float]:
    """Read the table of shortest distances shipped with metatrain.

    :return: A dictionary mapping ``(Z1, Z2)`` with ``Z1 <= Z2`` to the shortest
        distance (in Å) between the two elements in the training data.
    """
    table: Dict[Tuple[int, int], float] = {}
    with open(_MIN_DISTANCES_FILE, "r") as fd:
        for line in fd:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            z1, z2 = sorted((int(fields[0]), int(fields[1])))
            table[(z1, z2)] = float(fields[2])
    return table


def _check_atomic_numbers(z1: int, z2: int) -> None:
    for z in (z1, z2):
        if z < 1 or z > _MAX_Z:
            raise ValueError(
                f"the NLH potential is only available for elements with 1 <= Z <= "
                f"{_MAX_Z}, got Z={z}"
            )


def nlh_min_distance(z1: int, z2: int) -> float:
    """
    Shortest distance (in Å) between two atoms of a pair of elements in the training
    data of PET-MAD 2.0.

    This is the shortest of all interatomic distances (also between periodic images)
    in the training splits of the r2SCAN MAD 2.0 dataset (version 2.0rc5) and of the
    PBE MAD 1.5 dataset, which contain every pair of elements up to nobelium.

    :param z1: Atomic number of the first element.
    :param z2: Atomic number of the second element.
    :return: The shortest distance of the pair.
    """
    _check_atomic_numbers(z1, z2)
    return _tabulated_min_distances()[(min(z1, z2), max(z1, z2))]


def nlh_switching_radii(z1: int, z2: int) -> Tuple[float, float]:
    """
    Distances (in Å) between which the NLH potential of a pair is switched off.

    The potential vanishes above :math:`r_c = \\max(r_{min} - 0.4, r_{min} / 2)`,
    where :math:`r_{min}` is the shortest distance between the two elements in the
    training data (see :py:func:`nlh_min_distance`), so that it is zero for all of
    the training data. It is used unmodified below :math:`r_{on} = \\max(r_c - 0.4,
    0)`.

    :param z1: Atomic number of the first element.
    :param z2: Atomic number of the second element.
    :return: The onset :math:`r_{on}` and the cutoff :math:`r_c` of the switching
        function of the pair.
    """
    r_min = nlh_min_distance(z1, z2)
    r_cut = max(r_min - _CUTOFF_MARGIN, 0.5 * r_min)
    r_on = max(r_cut - _SWITCH_WIDTH, 0.0)
    return r_on, r_cut


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

    The potential is a safety wall below the training data. For every pair of
    elements, it is multiplied by a smooth (:math:`C^2`) switching function, going
    from one at :math:`r_{on}` to zero at :math:`r_c`, about 0.4 Å below the shortest
    distance between the two elements in the training data of PET-MAD 2.0 (see
    :py:func:`nlh_switching_radii`). It is therefore zero for all of the training
    data, which the machine learning model learns unchanged, including its
    short-range repulsion. The potential only acts on closer contacts, where the
    machine learning model has no data and its repulsion can break down.

    :param hypers: A dictionary of model hyperparameters. This parameter is ignored and
        is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    __checkpoint_version__ = 2
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    species_to_index: torch.Tensor
    coefficients_a: torch.Tensor
    coefficients_b: torch.Tensor
    switching_onsets: torch.Tensor
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
        self.coulomb_constant = _COULOMB_CONSTANT

        self.register_buffer("species_to_index", torch.empty(0, dtype=torch.int))
        self.register_buffer("coefficients_a", torch.empty(0, dtype=torch.float64))
        self.register_buffer("coefficients_b", torch.empty(0, dtype=torch.float64))
        self.register_buffer("switching_onsets", torch.empty(0, dtype=torch.float64))
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
        switching_onsets = torch.zeros((n_types, n_types), dtype=torch.float64)
        switching_radii = torch.zeros((n_types, n_types), dtype=torch.float64)

        for i, z_i in enumerate(self.atomic_types):
            species_to_index[z_i] = i
            for j, z_j in enumerate(self.atomic_types):
                a, b = nlh_coefficients(z_i, z_j)
                coefficients_a[i, j] = torch.tensor(a, dtype=torch.float64)
                coefficients_b[i, j] = torch.tensor(b, dtype=torch.float64)
                r_on, r_cut = nlh_switching_radii(z_i, z_j)
                switching_onsets[i, j] = r_on
                switching_radii[i, j] = r_cut

        if any(z > _MAX_TABULATED_Z for z in self.atomic_types):
            logging.info(
                "NLH coefficients for elements beyond uranium (Z > 92) are "
                "extrapolated from the uranium ones."
            )

        self.species_to_index = species_to_index.to(device)
        self.coefficients_a = coefficients_a.to(device)
        self.coefficients_b = coefficients_b.to(device)
        self.switching_onsets = switching_onsets.to(device)
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
        return self.pair_energy_and_derivative(zi, zj, rij)[0]

    def pair_energy_and_derivative(
        self, zi: torch.Tensor, zj: torch.Tensor, rij: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Switched NLH energy of isolated atom pairs and its derivative with respect to
        the distance, computed analytically.

        This is the only place where the potential is evaluated: the energy is used by
        :py:meth:`forward` (where autograd gives the forces and the stress), and the
        derivative to remove the potential from the targets, in the forked
        ``DataLoader`` workers where autograd can not run.

        :param zi: Atomic numbers of the first atoms.
        :param zj: Atomic numbers of the second atoms.
        :param rij: Distances between the atoms, in Å.
        :return: The energy (in eV) of each pair, and its derivative (in eV/Å).
        """
        ti = self.species_to_index[zi]
        tj = self.species_to_index[zj]
        a = self.coefficients_a[ti, tj].to(rij.dtype)
        b = self.coefficients_b[ti, tj].to(rij.dtype)
        r_on = self.switching_onsets[ti, tj].to(rij.dtype)
        r_cut = self.switching_radii[ti, tj].to(rij.dtype)

        exponentials = a * torch.exp(-b * rij.unsqueeze(-1))
        phi = torch.sum(exponentials, dim=-1)
        dphi = -torch.sum(b * exponentials, dim=-1)
        prefactor = self.coulomb_constant * (zi * zj).to(rij.dtype)
        energy = prefactor * phi / rij
        denergy = prefactor * (dphi / rij - phi / rij**2)

        # the derivative of the smootherstep, 30 x^2 (1 - x)^2, vanishes at the
        # edges of the clamp, so that it is also correct outside of [r_on, r_cut]
        x = torch.clamp((rij - r_on) / (r_cut - r_on), 0.0, 1.0)
        switch = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        dswitch = -30.0 * x**2 * (1.0 - x) ** 2 / (r_cut - r_on)

        return energy * switch, denergy * switch + energy * dswitch

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
        # the radii of the checkpoint can differ from the current tabulated ones
        model.cutoff_radius = float(torch.max(model.switching_radii))
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
        if checkpoint["model_ckpt_version"] == 1 and version >= 2:
            # Version 1 switched the potential on below a fixed fraction (0.673) of
            # per-pair radii, which are stored in the checkpoint: keep that potential
            for key in ["model_state_dict", "best_model_state_dict"]:
                state_dict = checkpoint.get(key)
                if state_dict is not None and "switching_onsets" not in state_dict:
                    state_dict["switching_onsets"] = (
                        0.673 * state_dict["switching_radii"]
                    )
            checkpoint["model_ckpt_version"] = 2
        if checkpoint["model_ckpt_version"] != version:
            raise NotImplementedError(
                f"Unable to upgrade the NLH checkpoint from version "
                f"{checkpoint['model_ckpt_version']} to version {version}."
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
