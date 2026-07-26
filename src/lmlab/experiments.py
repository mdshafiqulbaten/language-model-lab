from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    code_revision: str
    data_revision: str
    model_config: dict
    seed: int
    environment: dict


def manifest_checksum(manifest: ExperimentManifest) -> str:
    payload = json.dumps(
        asdict(manifest), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Variant:
    name: str
    components: dict[str, bool]


def validate_single_component_ablations(
    full: Variant, ablations: tuple[Variant, ...]
) -> bool:
    for ablation in ablations:
        keys = set(full.components) | set(ablation.components)
        differences = [
            key for key in keys if full.components.get(key) != ablation.components.get(key)
        ]
        if len(differences) != 1:
            raise ValueError(f"{ablation.name} changes {len(differences)} components")
    return True

