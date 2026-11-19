from typing import Literal, Optional, Union

from ..data import DatasetInfo
from .nlh import NLH
from .zbl import ZBL


RepulsionType = Optional[Literal["zbl", "nlh"]]
"""Short-range repulsion that can be used as an additive model: ``"zbl"`` for
:py:class:`ZBL`, ``"nlh"`` for :py:class:`NLH`, or ``None`` for no repulsion."""

_REPULSION_MODELS = {"zbl": ZBL, "nlh": NLH}


def get_repulsion_model(
    repulsion: Optional[str], dataset_info: DatasetInfo
) -> Optional[Union[ZBL, NLH]]:
    """
    Create the short-range repulsion additive model selected by an architecture's
    ``repulsion`` hyperparameter.

    Only the targets the repulsion model supports (scalar energies in eV) are kept
    from ``dataset_info``.

    :param repulsion: Name of the repulsion model, ``"zbl"`` or ``"nlh"``, or
        ``None`` to not use any.
    :param dataset_info: Information about the dataset the model is trained on.
    :return: The repulsion model, or ``None`` if ``repulsion`` is ``None``.
    """
    if repulsion is None:
        return None
    if repulsion not in _REPULSION_MODELS:
        raise ValueError(
            f"unknown repulsion '{repulsion}', expected one of "
            f"{sorted(_REPULSION_MODELS)} or None"
        )

    model_class = _REPULSION_MODELS[repulsion]
    targets = {
        target_name: target_info
        for target_name, target_info in dataset_info.targets.items()
        if model_class.is_valid_target(target_name, target_info)
    }
    return model_class(
        {},
        dataset_info=DatasetInfo(
            length_unit=dataset_info.length_unit,
            atomic_types=dataset_info.atomic_types,
            targets=targets,
        ),
    )
