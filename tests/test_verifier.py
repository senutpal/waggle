from __future__ import annotations


def test_low_tier_skipped():
    from src.verifier import verify_draft

    r = verify_draft("q", [], {"body": "hi"}, [], tier="low")
    assert r.verdict == "SKIPPED"


def test_citation_reject_local():
    from src.verifier import verify_draft

    draft = {"body": "See link.", "evidence_ids": ["D9"]}
    r = verify_draft("q", [{"id": "D1", "text": "x"}], draft, ["D1"], tier="high")
    assert r.verdict == "VERIFIER_REJECT" and r.failed_check == "citation"


def test_entity_reject_local():
    from src.verifier import verify_draft

    draft = {"body": "Call 1-800-111-2222.", "evidence_ids": ["D1"]}
    r = verify_draft("q", [{"id": "D1", "text": "Call 1-800-555-0147."}], draft, ["D1"], tier="high")
    assert r.verdict == "VERIFIER_REJECT" and r.failed_check == "entity"


def test_no_key_skipped_after_local_pass():
    from src.verifier import verify_draft

    draft = {"body": "Sorry \u2014 please DM @Support with details.", "evidence_ids": []}
    r = verify_draft("q", [{"id": "D1", "text": "x"}], draft, ["D1"], tier="high")
    assert r.verdict == "SKIPPED"


def test_injected_verdicts():
    from src.verifier import verify_draft

    assert verify_draft("q", [], {}, [], tier="high", _inject="VERIFIER_ACCEPT").verdict == "VERIFIER_ACCEPT"
    assert verify_draft("q", [], {}, [], tier="high", _inject="VERIFIER_REJECT").verdict == "VERIFIER_REJECT"
