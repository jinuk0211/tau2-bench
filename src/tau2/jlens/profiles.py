"""Pinned model/lens pairs used by the tau2 J-Lens analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class JLensModelProfile:
    """One immutable Hugging Face model and fitted Jacobian lens pair."""

    name: str
    model_id: str
    model_revision: str
    lens_repo: str
    lens_revision: str
    lens_file: str
    max_context_tokens: int = 32768
    min_vram_gib: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


PROFILES = {
    "qwen3-8b": JLensModelProfile(
        name="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="91271eb5b15a43eebed7bb447618738754f1379a",
        lens_file=(
            "qwen3-8b/jlens/Salesforce-wikitext/"
            "Qwen3-8B_jacobian_lens.pt"
        ),
        min_vram_gib=20.0,
    ),
    "qwen3.5-4b": JLensModelProfile(
        name="qwen3.5-4b",
        model_id="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="a4114d7752d11eb546e6cf372213d7e75526d3a1",
        lens_file=(
            "qwen3.5-4b/jlens/Salesforce-wikitext/"
            "Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
        min_vram_gib=12.0,
    ),
    "qwen3.5-9b-base": JLensModelProfile(
        name="qwen3.5-9b-base",
        model_id="Qwen/Qwen3.5-9B-Base",
        model_revision="68c46c4b3498877f3ef123c856ecfde50c39f404",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="a4114d7752d11eb546e6cf372213d7e75526d3a1",
        lens_file=(
            "qwen3.5-9b-pt/jlens/Salesforce-wikitext/"
            "Qwen3.5-9B-Base_jacobian_lens.pt"
        ),
        min_vram_gib=24.0,
    ),
    "qwen3.6-27b": JLensModelProfile(
        name="qwen3.6-27b",
        model_id="Qwen/Qwen3.6-27B",
        model_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="a4114d7752d11eb546e6cf372213d7e75526d3a1",
        lens_file=(
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        ),
        min_vram_gib=70.0,
    ),
}


def get_profile(name: str) -> JLensModelProfile:
    """Return a pinned profile or raise a useful configuration error."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown J-Lens profile {name!r}; choose one of {sorted(PROFILES)}"
        ) from exc
