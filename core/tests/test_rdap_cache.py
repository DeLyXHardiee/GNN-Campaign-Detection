import json

from core.preprocessing import RDAP_processor


def test_load_cache_ignores_lock_file_when_reading(tmp_path, monkeypatch):
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()

    cache_file = cache_dir / "rdap_cache.json"
    cache_file.write_text(json.dumps({"example.com": {"registrar": "Example Registrar"}}), encoding="utf-8")

    lock_file = cache_dir / "rdap_cache.json.lock"
    lock_file.write_text("12345", encoding="ascii")

    monkeypatch.setattr(RDAP_processor, "RDAP_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(RDAP_processor, "RDAP_CACHE_FILE", str(cache_file))
    monkeypatch.setattr(RDAP_processor, "RDAP_CACHE_LOCK_FILE", str(lock_file))
    monkeypatch.setattr(RDAP_processor, "LEGACY_RDAP_CACHE_FILE", str(tmp_path / "legacy_rdap_cache.json"))

    cache = RDAP_processor.load_cache()

    assert cache == {"example.com": {"registrar": "Example Registrar"}}