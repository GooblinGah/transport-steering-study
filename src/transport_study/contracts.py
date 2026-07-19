from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class TaskManifest:
    donor: str
    cytokine: str
    heldout_class: str
    p0: tuple[str, ...]
    p1: tuple[str, ...]
    q0: tuple[str, ...]
    y1: tuple[str, ...]
    source_class_counts: dict[str, dict[str, int]]
    genes: tuple[str, ...]
    seed: int
    resample: int
    fine_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    sampling_mode: str = "fine_matched"
    eligible: bool = True
    eligibility_reason: str = ""
    preprocessing: dict[str, Any] = field(default_factory=lambda: {"input": "raw_counts"})

    @property
    def task_id(self) -> str:
        parts=(self.donor,self.cytokine,self.heldout_class,self.sampling_mode)
        return "__".join(quote(str(x),safe="-._") for x in parts)+f"__r{self.resample:02d}"

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.payload())

    def validate(self, minimum: int = 32) -> None:  # floor = smallest amended role tier (see docs/AMENDMENT_ROLE_SIZE.md)
        roles = {"p0": self.p0, "p1": self.p1, "q0": self.q0, "y1": self.y1}
        for name, ids in roles.items():
            if self.eligible and len(ids) < minimum:
                raise ValueError(f"{self.task_id}: {name} has {len(ids)} cells (<{minimum})")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{self.task_id}: duplicate IDs in {name}")
        names = list(roles)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = set(roles[left]) & set(roles[right])
                if overlap:
                    raise ValueError(f"{self.task_id}: {left}/{right} overlap ({len(overlap)})")
        if len(self.genes) != len(set(self.genes)):
            raise ValueError("gene order contains duplicates")
        if self.eligible and self.sampling_mode == "fine_matched":
            for left, right in (("p0", "p1"), ("q0", "y1")):
                if self.fine_label_counts.get(left) != self.fine_label_counts.get(right):
                    raise ValueError(f"{self.task_id}: fine-label counts differ for {left}/{right}")

    def write(self, path: Path) -> None:
        self.validate()
        body = self.payload() | {"task_id": self.task_id, "manifest_hash": self.manifest_hash}
        path.parent.mkdir(parents=True, exist_ok=True)
        content=json.dumps(body, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text()!=content:raise FileExistsError(f"immutable manifest already exists with different content: {path}")
            return
        path.write_text(content)


def validate_manifest_payload(body: dict[str, Any], minimum: int = 32) -> TaskManifest:
    """Rebuild and authenticate a manifest read from mutable JSON."""
    payload = {key: body[key] for key in TaskManifest.__dataclass_fields__}
    for key in ("p0", "p1", "q0", "y1", "genes"):
        payload[key] = tuple(payload[key])
    manifest = TaskManifest(**payload)
    manifest.validate(minimum=minimum)
    if body.get("task_id") != manifest.task_id:
        raise ValueError("manifest task_id does not match its payload")
    if body.get("manifest_hash") != manifest.manifest_hash:
        raise ValueError("manifest hash does not match its payload")
    return manifest


@dataclass(frozen=True)
class MethodConfig:
    name: str
    representation: str
    operator: str
    layer: int | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    stack_revision: str = "cacc2e4b09435c3e536d46237d10b50f222dd144"
    model_revision: str = "b09f085dac03d170b078a5c72f550ae93686e544"
    decoder_policy: str = "query_control_library_size"
    comparison_table: str = "end_to_end"
    decoder_family: str = "native"
    decoder_train_ids_hash: str = ""
    expression_basis_hash: str = ""
    transport_config_hash: str = ""
    window_seed: int = 0
    ridge_grid: tuple[float, ...] = ()
    selected_ridge: float | None = None
    deterministic_mean: bool = True
    output_scale: str = "expected_counts"
    correction_policy: str = "synthetic_control"

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class WindowConfig:
    """Define a Stack cell context. Each window record contains its cells and order."""
    stage: str
    window_size: int = 512
    focal_cells: int = 512
    context_cells: int = 0
    query_cells: int = 0
    p0_p1_encoding: str = "separate_matched_windows"
    q0_present_during_source_fit: bool = False
    resample_composition: bool = True
    resample_order: bool = True
    prompt_ratio: float | None = None
    context_ratio: float | None = None
    num_steps: int | None = None

    def validate(self) -> None:
        if self.focal_cells + self.context_cells + self.query_cells != self.window_size:
            raise ValueError("window components must sum to window_size")
        if self.stage == "source_fit" and (self.q0_present_during_source_fit or self.query_cells):
            raise ValueError("Q0/query cells are forbidden during source-map fitting")


@dataclass(frozen=True)
class PredictionBundle:
    task_id: str
    method: str
    perturbed_path: str
    noop_path: str
    query_cell_ids: tuple[str, ...]
    output_scale: str = "expected_counts"
    library_size_policy: str = "query_control_observed"
    deterministic_mean: bool = True
    synthetic_control_formula: str = "clip(Q0 + perturbed - noop, 0); rescale_to_Q0_library"
    manifest_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.output_scale != "expected_counts":
            raise ValueError("prediction bundles must contain expected counts")
        if not self.deterministic_mean:
            raise ValueError("primary bundles require deterministic means")


@dataclass(frozen=True)
class MetricRecord:
    task_id: str
    donor: str
    cytokine: str
    heldout_class: str
    method: str
    resample: int
    metric: str
    value: float | None
    eligible: bool
    eligibility_reason: str = ""
    evaluator_panel: str = "control_only_fixed"
    comparison_table: str = "end_to_end"
    generation_seed: int = 0
    manifest_hash: str = ""
    method_config_hash: str = ""
    panel_hash: str = ""
    evaluator_hash: str = ""
    deg_count: int | None = None
    deg_set_hash: str = ""
    decoder_train_ids_hash: str = ""
    expression_basis_hash: str = ""
    window_seed: int = 0
    transport_config_hash: str = ""
    sampling_mode: str = "fine_matched"
