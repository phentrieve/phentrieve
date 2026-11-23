"""Disease ranking router: flag, index availability, validation, research-use gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.config as api_config
from api import disease_index_state
from api.main import create_app
from tests.unit.disease_matching.conftest import build_mini

pytestmark = pytest.mark.unit

PROFILE = {
    "terms": [
        {"hpo_id": "HP:0000003", "assertion": "affirmed"},
        {"hpo_id": "HP:0000020", "assertion": "affirmed"},
        {"hpo_id": "HP:0000011", "assertion": "negated"},
        {"hpo_id": "HP:0000004", "assertion": "uncertain"},
    ]
}


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("disease_index")
    build_mini().save(path)
    return path


@pytest.fixture
def enabled_client(monkeypatch, index_dir):
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(index_dir))
    disease_index_state.reset_disease_index()
    yield TestClient(create_app())
    disease_index_state.reset_disease_index()


def test_router_not_mounted_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", False)
    client = TestClient(create_app())
    assert client.post("/api/v1/diseases/rank", json=PROFILE).status_code == 404


def test_rank_diseases(enabled_client) -> None:
    r = enabled_client.post("/api/v1/diseases/rank", json=PROFILE)
    assert r.status_code == 200
    body = r.json()
    assert body["level"] == "disease"
    assert body["score_kind"] == "raw_uncalibrated_phenotype_similarity"
    assert "not a diagnosis" in body["intended_use"]
    assert body["scored"] == ["HP:0000003", "HP:0000020"]
    assert body["excluded"] == ["HP:0000011"]
    assert body["not_scored"] == [{"hpo_id": "HP:0000004", "reason": "uncertain"}]
    assert body["total_candidates"] == 3
    assert body["genes"] == []
    top = body["diseases"][0]
    assert top["disease_id"] == "OMIM:100001"
    assert top["twins"] == ["ORPHA:900001"]
    assert top["genes"] == ["GENEA"]
    assert top["rank"] == 1
    assert {m["profile_id"] for m in top["explanation"]["matches"]} == {
        "HP:0000003",
        "HP:0000020",
    }
    assert top["explanation"]["conflicts"][0]["profile_id"] == "HP:0000011"
    assert body["index_info"]["hpo_version"] == "v2026-09-01"


def test_rank_genes_and_include_uncertain(enabled_client) -> None:
    r = enabled_client.post(
        "/api/v1/diseases/rank",
        json={**PROFILE, "level": "gene", "include_uncertain": True, "top_k": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["diseases"] == []
    assert len(body["genes"]) == 2
    assert body["genes"][0]["gene"] == "GENEA"
    assert body["genes"][0]["best_disease_id"] == "OMIM:100001"
    assert "HP:0000004" in body["scored"]


@pytest.mark.parametrize(
    "payload",
    [
        {"terms": []},
        {"terms": [{"hpo_id": "HP:123", "assertion": "affirmed"}]},
        {"terms": [{"hpo_id": "HP:0000003", "assertion": "maybe"}]},
        {"terms": [{"hpo_id": "HP:0000003"}], "level": "pathway"},
        {"terms": [{"hpo_id": "HP:0000003"}], "top_k": 101},
    ],
)
def test_invalid_requests_return_422(enabled_client, payload) -> None:
    assert enabled_client.post("/api/v1/diseases/rank", json=payload).status_code == 422


def test_missing_index_returns_503(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    disease_index_state.reset_disease_index()
    client = TestClient(create_app())
    r = client.post("/api/v1/diseases/rank", json=PROFILE)
    assert r.status_code == 503
    assert "disease_index_unavailable" in r.text
    disease_index_state.reset_disease_index()


def test_research_use_acknowledgement_is_required_in_hosted_mode(
    enabled_client, monkeypatch
) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_PUBLIC_HOSTED_MODE", True)
    r = enabled_client.post("/api/v1/diseases/rank", json=PROFILE)
    assert r.status_code == 428
    ok = enabled_client.post(
        "/api/v1/diseases/rank",
        json=PROFILE,
        headers={"X-Phentrieve-Research-Use-Acknowledged": "true"},
    )
    assert ok.status_code == 200


def test_status_reports_feature_and_index(monkeypatch, index_dir, tmp_path) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", False)
    disease_index_state.reset_disease_index()
    assert disease_index_state.disease_ranking_status() == {
        "enabled": False,
        "index_ready": False,
        "hpo_version": None,
        "mondo_version": None,
        "built_at": None,
    }
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    assert disease_index_state.disease_ranking_status()["index_ready"] is False
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(index_dir))
    status = disease_index_state.disease_ranking_status()
    assert status["enabled"] is True
    assert status["index_ready"] is True
    assert status["hpo_version"] == "v2026-09-01"


def test_config_info_includes_disease_ranking(enabled_client, mocker) -> None:
    mocker.patch(
        "api.routers.config_info_router.load_hpo_graph_data", return_value=({}, {})
    )
    r = enabled_client.get("/api/v1/info")
    assert r.status_code == 200, r.text
    status = r.json()["disease_ranking"]
    assert status["enabled"] is True
    assert status["index_ready"] is True


def test_failed_load_is_cached_and_reported_not_ready(
    monkeypatch, tmp_path, mocker
) -> None:
    from phentrieve.disease_matching.index import INDEX_ARRAYS_FILENAME

    build_mini().save(tmp_path)
    path = tmp_path / INDEX_ARRAYS_FILENAME
    path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    disease_index_state.reset_disease_index()
    loader = mocker.spy(disease_index_state, "load_index")
    client = TestClient(create_app())
    assert client.post("/api/v1/diseases/rank", json=PROFILE).status_code == 503
    assert client.post("/api/v1/diseases/rank", json=PROFILE).status_code == 503
    assert loader.call_count == 1  # second request fails fast
    assert disease_index_state.disease_ranking_status()["index_ready"] is False
    disease_index_state.reset_disease_index()


def test_status_rejects_an_unsupported_index_format(monkeypatch, tmp_path) -> None:
    import json

    from phentrieve.disease_matching.index import INDEX_INFO_FILENAME

    build_mini().save(tmp_path)
    info_path = tmp_path / INDEX_INFO_FILENAME
    info = json.loads(info_path.read_text())
    info["format_version"] = 999
    info_path.write_text(json.dumps(info))
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    disease_index_state.reset_disease_index()
    assert disease_index_state.disease_ranking_status()["index_ready"] is False


def test_expired_failure_is_retried_and_reported_ready(
    monkeypatch, tmp_path, mocker
) -> None:
    from phentrieve.disease_matching.index import INDEX_ARRAYS_FILENAME

    build_mini().save(tmp_path)
    path = tmp_path / INDEX_ARRAYS_FILENAME
    good = path.read_bytes()
    path.write_bytes(good + b"corrupt")
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    disease_index_state.reset_disease_index()
    client = TestClient(create_app())
    assert client.post("/api/v1/diseases/rank", json=PROFILE).status_code == 503
    assert disease_index_state.disease_ranking_status()["index_ready"] is False
    path.write_bytes(good)  # repaired
    monkeypatch.setattr(disease_index_state, "RETRY_AFTER_SECONDS", 0.0)
    assert disease_index_state.disease_ranking_status()["index_ready"] is True
    assert client.post("/api/v1/diseases/rank", json=PROFILE).status_code == 200
    disease_index_state.reset_disease_index()


def test_rebuilt_index_is_reloaded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_RANKING_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", str(tmp_path))
    disease_index_state.reset_disease_index()
    build_mini().save(tmp_path)
    first = disease_index_state.get_disease_index()
    assert disease_index_state.get_disease_index() is first  # cached
    rebuilt = build_mini(merge_orpha=False)
    rebuilt.save(tmp_path)
    second = disease_index_state.get_disease_index()
    assert second is not first
    assert second.meta["merge_orpha"] is False
    disease_index_state.reset_disease_index()


def test_index_dir_follows_the_cli_resolution(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_config, "PHENTRIEVE_DISEASE_INDEX_DIR", "")
    monkeypatch.setattr(
        "phentrieve.disease_matching.index.default_index_dir",
        lambda data_dir=None: tmp_path / "from-cli",
    )
    monkeypatch.setattr(
        disease_index_state,
        "default_index_dir",
        lambda data_dir=None: tmp_path / "from-cli",
    )
    assert disease_index_state.disease_index_dir() == tmp_path / "from-cli"
