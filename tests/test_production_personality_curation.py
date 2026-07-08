from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_production_personality_basis import (
    HARD_NEGATIVE_ANCHORS,
    compound_tail_fragments,
    curated_keep,
    hyphen_bundle_fragment,
    load_curation_rejections,
    semantic_keep,
)
from cache_contextual_personality_anchor_scores import NEGATIVE_ANCHORS


CURATION_PATH = ROOT / "config" / "personality_descriptor_curation.json"
CURRENT_BASIS_PATH = ROOT / "run" / "production_personality_basis" / "production_personality_basis.tsv"


def current_basis_rows() -> dict[str, dict[str, str]]:
    with CURRENT_BASIS_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["descriptor"]: row for row in csv.DictReader(handle, dialect="excel-tab")}


def test_curation_rejections_are_honored_by_current_basis() -> None:
    curated = load_curation_rejections(CURATION_PATH)
    rows = current_basis_rows()

    leaked = {
        descriptor
        for descriptor in curated
        if descriptor in rows and str(rows[descriptor].get("keep")).lower() == "true"
    }
    assert leaked == set()


def test_known_non_personality_leaks_are_not_bespoke_curation_entries() -> None:
    payload = json.loads(CURATION_PATH.read_text(encoding="utf-8"))
    rejects = payload["reject_descriptors"]

    assert "oriented" not in rejects
    assert "laid" not in rejects
    assert "daily" not in rejects
    assert "crippled" not in rejects


def test_negative_anchors_include_general_non_personality_classes() -> None:
    anchors = set(NEGATIVE_ANCHORS)

    assert "time frequency or schedule" in anchors
    assert "daily routine or ordinary activity" in anchors
    assert "physical disability or impairment" in anchors
    assert "incomplete relational adjective" in anchors
    assert "participle state fragment" in anchors
    assert "compound adjective tail fragment" in anchors
    assert "time frequency or schedule" in HARD_NEGATIVE_ANCHORS
    assert "physical disability or impairment" in HARD_NEGATIVE_ANCHORS
    assert "compound adjective tail fragment" in HARD_NEGATIVE_ANCHORS


def test_curated_keep_preserves_false_and_blocks_manual_rejects() -> None:
    rejection_reasons = {
        "previously-false": "locked-reject: false in accepted snapshot",
    }

    assert curated_keep(True, "previously-false", rejection_reasons) == (
        False,
        "locked-reject: false in accepted snapshot",
    )
    assert curated_keep(True, "tsundere", rejection_reasons) == (True, "")
    assert curated_keep(False, "tsundere", rejection_reasons) == (False, "")


def test_semantic_gate_uses_llm_decision_without_bespoke_word_rejects() -> None:
    decisions = {
        "kept-term": {"llm_keep": True, "llm_reason": "", "llm_model": "test"},
        "rejected-term": {"llm_keep": False, "llm_reason": "", "llm_model": "test"},
    }

    keep, details = semantic_keep(
        "kept-term",
        anchor_keep=False,
        best_negative_anchor="physical appearance",
        score_mean=0.0,
        llm_decisions=decisions,
    )
    assert keep is True
    assert details["decision_reason"] == "llm_decision"

    keep, details = semantic_keep(
        "rejected-term",
        anchor_keep=True,
        best_negative_anchor="personality-looking-noise",
        score_mean=0.05,
        llm_decisions=decisions,
    )
    assert keep is False
    assert details["decision_reason"] == "llm_reject"

    keep, details = semantic_keep(
        "kept-term",
        anchor_keep=True,
        best_negative_anchor="time frequency or schedule",
        score_mean=-0.01,
        llm_decisions=decisions,
    )
    assert keep is False
    assert details["decision_reason"] == "hard_negative_anchor"


def test_compound_tail_fragment_gate_blocks_bare_tail_without_bespoke_reject() -> None:
    payload = json.loads(CURATION_PATH.read_text(encoding="utf-8"))
    rejects = payload["reject_descriptors"]
    assert "spirited" not in rejects

    tail_fragments = compound_tail_fragments(["free-spirited", "high-spirited", "spirited", "tsundere"])
    assert "spirited" in tail_fragments
    assert "free-spirited" not in tail_fragments
    assert "high-spirited" not in tail_fragments

    decisions = {
        "spirited": {"llm_keep": True, "llm_reason": "", "llm_model": "test"},
        "free-spirited": {"llm_keep": True, "llm_reason": "", "llm_model": "test"},
    }

    keep, details = semantic_keep(
        "spirited",
        anchor_keep=True,
        best_negative_anchor="personality adjective",
        score_mean=0.05,
        llm_decisions=decisions,
        tail_fragments=tail_fragments,
    )
    assert keep is False
    assert details["decision_reason"] == "compound_tail_fragment"

    keep, details = semantic_keep(
        "free-spirited",
        anchor_keep=True,
        best_negative_anchor="personality adjective",
        score_mean=0.05,
        llm_decisions=decisions,
        tail_fragments=tail_fragments,
    )
    assert keep is True
    assert details["decision_reason"] == "llm_decision"


def test_hyphen_bundle_gate_blocks_glued_descriptor_lists() -> None:
    assert hyphen_bundle_fragment("tsundere-sadistic-bossy")
    assert hyphen_bundle_fragment("happy-go-lucky")
    assert not hyphen_bundle_fragment("sharp-tongued")
    assert not hyphen_bundle_fragment("free-spirited")

    decisions = {
        "tsundere-sadistic-bossy": {"llm_keep": True, "llm_reason": "", "llm_model": "test"},
        "sharp-tongued": {"llm_keep": True, "llm_reason": "", "llm_model": "test"},
    }

    keep, details = semantic_keep(
        "tsundere-sadistic-bossy",
        anchor_keep=True,
        best_negative_anchor="behavioral archetype",
        score_mean=0.05,
        llm_decisions=decisions,
    )
    assert keep is False
    assert details["decision_reason"] == "hyphen_bundle_fragment"

    keep, details = semantic_keep(
        "sharp-tongued",
        anchor_keep=True,
        best_negative_anchor="behavioral archetype",
        score_mean=0.05,
        llm_decisions=decisions,
    )
    assert keep is True
    assert details["decision_reason"] == "llm_decision"
