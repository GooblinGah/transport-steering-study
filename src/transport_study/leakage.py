"""Record fit artifacts. Reject an artifact that uses sealed Y1 cells."""
from __future__ import annotations
from dataclasses import dataclass,field
from .contracts import TaskManifest,canonical_hash


@dataclass
class LeakageGuard:
    manifest: TaskManifest
    artifacts: dict[str,dict]=field(default_factory=dict)

    def register_fit(self,name,cell_ids,**provenance):
        ids=tuple(map(str,cell_ids));overlap=set(ids)&set(self.manifest.y1)
        if overlap:raise ValueError(f"{name} overlaps sealed Y1 ({len(overlap)} cells)")
        if name in self.artifacts:raise ValueError(f"artifact {name!r} already registered")
        record={"cell_ids_hash":canonical_hash(ids),"n_cells":len(ids),"manifest_hash":self.manifest.manifest_hash}|provenance
        self.artifacts[name]=record;return record

    def validate_method_roles(self,role_ids):
        expected={"p0":self.manifest.p0,"p1":self.manifest.p1,"q0":self.manifest.q0,"y1":self.manifest.y1}
        normalized={k:tuple(map(str,v)) for k,v in role_ids.items()}
        if normalized!=expected:raise ValueError("method roles differ from immutable task manifest")

    @property
    def registry_hash(self):return canonical_hash(self.artifacts)
