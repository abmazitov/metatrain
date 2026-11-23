import math

import metatensor.torch as mts
import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive import NLH
from metatrain.utils.additive.nlh import (
    _COULOMB_CONSTANT,
    _SWITCH_ONSET,
    nlh_coefficients,
    nlh_switching_radius,
)
from metatrain.utils.additive.remove import _evaluate_pair_potential
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import arch_model_from_checkpoint
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

# rows of the published table (Z1, Z2, a1, b1, a2, b2, a3, b3)
PUBLISHED = [
    (1, 1, -8.99999, 9.55658, 9.99999, 8.89086, 0.0, 0.0),
    (8, 92, 0.44835, 20.22616, 0.55165, 4.56252, 0.0, 0.0),
    (26, 26, 0.34794, 19.25771, 0.65206, 4.81918, 0.0, 0.0),
    (79, 79, 0.20463, 44.96633, 0.63053, 8.73611, 0.16484, 3.45776),
]


def _dataset_info(atomic_types, length_unit="angstrom") -> DatasetInfo:
    return DatasetInfo(
        length_unit=length_unit,
        atomic_types=atomic_types,
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )


def _dimer(z1: int, z2: int, r: float, dtype=torch.float64) -> System:
    return System(
        types=torch.tensor([z1, z2], dtype=torch.int32),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, r]], dtype=dtype),
        cell=torch.zeros((3, 3), dtype=dtype),
        pbc=torch.tensor([False, False, False]),
    )


def _energy(model, systems, sample_kind="system"):
    nls = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(s, nls) for s in systems]
    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", sample_kind=sample_kind)
    }
    return model(systems, outputs)["energy"].block().values


def _raw_nlh(z1, z2, r, a, b):
    phi = sum(a_k * math.exp(-b_k * r) for a_k, b_k in zip(a, b, strict=True))
    return _COULOMB_CONSTANT * z1 * z2 / r * phi


@pytest.mark.parametrize("row", PUBLISHED)
def test_nlh_reproduces_published_potential(row):
    """Below the switching onset, the energy of a dimer is the published NLH one."""
    z1, z2, a1, b1, a2, b2, a3, b3 = row
    a, b = nlh_coefficients(z1, z2)
    assert a == pytest.approx([a1, a2, a3])
    assert b == pytest.approx([b1, b2, b3])

    model = NLH({}, _dataset_info(sorted({z1, z2})))
    r_on = _SWITCH_ONSET * nlh_switching_radius(z1, z2)
    for r in [0.05, 0.3 * r_on, 0.9 * r_on]:
        energy = _energy(model, [_dimer(z1, z2, r)])
        expected = _raw_nlh(z1, z2, r, [a1, a2, a3], [b1, b2, b3])
        assert energy.item() == pytest.approx(expected, rel=1e-10)


def test_nlh_extrapolation_beyond_uranium():
    """Elements heavier than U borrow the U coefficients with rescaled exponents."""
    a_ref, b_ref = nlh_coefficients(26, 92)
    a, b = nlh_coefficients(95, 26)

    def screening_length(z1, z2):
        return 0.46850 / (z1**0.23 + z2**0.23)

    scale = screening_length(26, 92) / screening_length(26, 95)
    assert a == a_ref
    assert b == pytest.approx([b_k * scale for b_k in b_ref])

    # every element up to nobelium is supported
    model = NLH({}, _dataset_info([1, 8, 26, 95, 102]))
    energy = _energy(model, [_dimer(102, 102, 1.0), _dimer(95, 1, 0.5)])
    assert torch.all(torch.isfinite(energy))
    assert torch.all(energy > 0)


def test_nlh_rejects_unknown_elements():
    with pytest.raises(ValueError, match="1 <= Z <= 102"):
        NLH({}, _dataset_info([1, 103]))


@pytest.mark.parametrize("pair", [(1, 1), (1, 8), (6, 6), (26, 26), (79, 8), (98, 6)])
def test_nlh_is_smooth_and_repulsive(pair):
    """Energy is positive and decreasing, and vanishes smoothly at the cutoff."""
    z1, z2 = pair
    model = NLH({}, _dataset_info(sorted({z1, z2})))
    r_cut = nlh_switching_radius(z1, z2)
    assert model.cutoff_radius >= r_cut

    r = torch.linspace(0.05, 1.2 * r_cut, 2000, dtype=torch.float64)
    r.requires_grad_(True)
    zi = torch.full_like(r, z1, dtype=torch.int32)
    zj = torch.full_like(r, z2, dtype=torch.int32)
    energy = model.pair_energy(zi, zj, r)
    (force,) = torch.autograd.grad(-energy.sum(), r)
    energy = energy.detach()

    inside = r.detach() < r_cut
    assert torch.all(energy[inside] > 0)
    assert torch.all(force[inside] >= 0)
    assert torch.all(energy[~inside] == 0)
    assert torch.all(force[~inside] == 0)

    # C1 continuity at the cutoff
    eps = 1e-6
    near = torch.tensor([r_cut - eps], dtype=torch.float64, requires_grad=True)
    e_near = model.pair_energy(zi[:1], zj[:1], near)
    (f_near,) = torch.autograd.grad(-e_near.sum(), near)
    assert e_near.item() == pytest.approx(0.0, abs=1e-9)
    assert f_near.item() == pytest.approx(0.0, abs=1e-6)


def test_nlh_forward():
    model = NLH({}, _dataset_info([1, 6, 8]))
    systems = [
        _dimer(1, 1, 0.3),
        _dimer(6, 8, 0.6),
        System(
            types=torch.tensor([6, 1, 1, 8], dtype=torch.int32),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.5]],
                dtype=torch.float64,
            ),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        ),
    ]

    per_system = _energy(model, systems)
    assert per_system.shape == (3, 1)

    # the dimer energy is the pair energy
    expected = model.pair_energy(
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([0.6], dtype=torch.float64),
    )
    assert torch.allclose(per_system[1], expected)

    # per-atom energies sum to the per-system energy, and batching is irrelevant
    per_atom = _energy(model, systems, sample_kind="atom")
    assert per_atom.shape == (8, 1)
    assert torch.allclose(per_atom[4:].sum(), per_system[2])
    assert torch.allclose(_energy(model, [systems[2]]), per_system[2])

    # selected atoms
    nls = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(s, nls) for s in systems]
    selected_atoms = mts.Labels(["system", "atom"], torch.tensor([[2, 0], [2, 3]]))
    output = model(
        systems,
        {"energy": ModelOutput(quantity="energy", unit="eV", sample_kind="atom")},
        selected_atoms=selected_atoms,
    )
    assert output["energy"].block().values.shape == (2, 1)


def test_nlh_torchscript():
    model = NLH({}, _dataset_info([1, 6, 8]))
    systems = [_dimer(1, 6, 0.4, dtype=torch.float32), _dimer(8, 8, 0.7, torch.float32)]
    expected = _energy(model, systems)
    scripted = torch.jit.script(model)
    assert torch.allclose(_energy(scripted, systems), expected)


def test_nlh_checkpoint_round_trip():
    model = NLH({}, _dataset_info([1, 26]))
    systems = [_dimer(1, 26, 0.5), _dimer(26, 26, 1.2)]
    expected = _energy(model, systems)

    loaded = arch_model_from_checkpoint(model.get_checkpoint(), context="export")
    assert isinstance(loaded, NLH)
    assert torch.allclose(_energy(loaded, systems), expected)


def test_nlh_restart_with_new_types():
    model = NLH({}, _dataset_info([1, 6]))
    model.restart(_dataset_info([1, 6, 26]))
    assert model.atomic_types == [1, 6, 26]
    energy = _energy(model, [_dimer(26, 6, 0.7)])
    assert energy.item() > 0


@pytest.mark.parametrize("length_unit", ["nanometer", "bohr", ""])
def test_nlh_rejects_lengths_that_are_not_angstroms(length_unit):
    with pytest.raises(ValueError, match="NLH only supports angstrom"):
        NLH({}, _dataset_info([1, 6], length_unit))


def test_nlh_analytic_gradients_match_autograd():
    """The gradients used to remove NLH from the targets (without autograd, see
    ``_evaluate_pair_potential``) are the ones of autograd."""
    torch.manual_seed(0)
    types = [1, 6, 8, 26, 79]
    target_info = get_energy_target_info(
        "energy",
        {"unit": "eV"},
        add_position_gradients=True,
        add_strain_gradients=True,
    )
    model = NLH({}, _dataset_info(types))

    systems = [_dimer(6, 6, 0.6), _dimer(1, 8, 0.9)]
    for _ in range(3):
        # periodic systems, with short distances, also between periodic images
        n_atoms = 6
        cell = 3.0 * torch.eye(3, dtype=torch.float64) + 0.3 * torch.rand(
            3, 3, dtype=torch.float64
        )
        systems.append(
            System(
                types=torch.tensor(types)[torch.randint(len(types), (n_atoms,))].to(
                    torch.int32
                ),
                positions=torch.rand(n_atoms, 3, dtype=torch.float64) @ cell,
                cell=cell,
                pbc=torch.tensor([True, True, True]),
            )
        )
    nls = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(s, nls) for s in systems]

    reference = evaluate_model(
        model, systems, {"energy": target_info}, is_training=False
    )["energy"].block()
    analytic = _evaluate_pair_potential(model, systems, {"energy": target_info})[
        "energy"
    ].block()

    assert torch.allclose(analytic.values, reference.values)
    for parameter in ["positions", "strain"]:
        assert reference.gradient(parameter).values.abs().max() > 0
        assert mts.equal_metadata_block(
            analytic.gradient(parameter), reference.gradient(parameter)
        )
        assert torch.allclose(
            analytic.gradient(parameter).values, reference.gradient(parameter).values
        )
