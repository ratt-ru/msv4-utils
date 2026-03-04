"""Tests for msv4_utils.uri."""

import pytest

from msv4_utils.uri import MSv4Backend, check_remote_zarr, check_rdb_magic, infer_backend


# ---------------------------------------------------------------------------
# CASA Table
# ---------------------------------------------------------------------------


def test_casa_table_detected(tmp_path):
    ms = tmp_path / "test.ms"
    ms.mkdir()
    (ms / "table.dat").write_bytes(b"\xbe\xbe\xbe\xbe" + b"\x00" * 100)
    assert infer_backend(str(ms)) == MSv4Backend.CASA_TABLE


def test_casa_table_wrong_magic(tmp_path):
    ms = tmp_path / "test.ms"
    ms.mkdir()
    (ms / "table.dat").write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
    assert infer_backend(str(ms)) != MSv4Backend.CASA_TABLE


def test_casa_table_no_table_dat(tmp_path):
    ms = tmp_path / "test.ms"
    ms.mkdir()
    assert infer_backend(str(ms)) != MSv4Backend.CASA_TABLE


# ---------------------------------------------------------------------------
# Zarr (local)
# ---------------------------------------------------------------------------


def test_zarr_v2_detected(tmp_path):
    store = tmp_path / "test_v2"
    store.mkdir()
    (store / ".zattrs").write_text("{}")
    assert infer_backend(str(store)) == MSv4Backend.ZARR


def test_zarr_v3_detected(tmp_path):
    store = tmp_path / "test_v3"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    assert infer_backend(str(store)) == MSv4Backend.ZARR


def test_zarr_neither_marker(tmp_path):
    store = tmp_path / "test_empty"
    store.mkdir()
    assert infer_backend(str(store)) == MSv4Backend.UNKNOWN


# ---------------------------------------------------------------------------
# MeerKAT Archive
# ---------------------------------------------------------------------------


def test_meerkat_http():
    uri = "http://archive-gw-1.kat.ac.za/obs/1234567890.rdb"
    assert infer_backend(uri) == MSv4Backend.MEERKAT


def test_meerkat_https_with_token():
    uri = "https://archive-gw-1.kat.ac.za/obs/1234567890.rdb?token=eyJhbGc.abc.xyz"
    assert infer_backend(uri) == MSv4Backend.MEERKAT


def test_meerkat_https_no_token():
    uri = "https://archive-gw-1.kat.ac.za/obs/1234567890.rdb"
    assert infer_backend(uri) == MSv4Backend.MEERKAT


def test_meerkat_https_no_rdb(httpserver):
    # No .rdb extension -> not MeerKAT; both Zarr markers return 404 -> UNKNOWN
    httpserver.expect_request("/obs/data.zarr/.zattrs", method="HEAD").respond_with_data("", status=404)
    httpserver.expect_request("/obs/data.zarr/zarr.json", method="HEAD").respond_with_data("", status=404)
    # Assume it's Zarr in non-strict mode
    assert infer_backend(httpserver.url_for("/obs/data.zarr")) == MSv4Backend.ZARR
    # Do not assume it's zarr in strict mode
    assert infer_backend(httpserver.url_for("/obs/data.zarr"), strict=True) == MSv4Backend.UNKNOWN


# ---------------------------------------------------------------------------
# MeerKAT strict mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_meerkat_strict_redis_magic_matches(httpserver, monkeypatch, scheme):
    path = "/obs/1234567890.rdb"
    httpserver.expect_request(path).respond_with_data(b"REDIS0006", status=200)
    actual_url = f"{scheme}://archive-gw-1.kat.ac.za/{path}"
    rdb_url = httpserver.url_for(path)
    # Substitute the mock server url for the actual url as the mock server cannot access it
    monkeypatch.setattr("msv4_utils.uri.check_rdb_magic", lambda _: check_rdb_magic(rdb_url))
    assert infer_backend(actual_url, strict=True) == MSv4Backend.MEERKAT


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_meerkat_strict_redis_magic_no_match(httpserver, monkeypatch, scheme):
    path = "/obs/1234567890.rdb"
    httpserver.expect_request(path).respond_with_data(b"\x00\x00\x00\x00\x00", status=200)
    real_url = f"{scheme}://archive-gw-1.kat.ac.za/{path}"
    actual_url = httpserver.url_for(path)
    # Substitute a mock server url for the actual url
    monkeypatch.setattr("msv4_utils.uri.check_rdb_magic", lambda _: check_rdb_magic(actual_url))
    assert infer_backend(real_url, strict=True) == MSv4Backend.UNKNOWN


# ---------------------------------------------------------------------------
# Remote Zarr — non-http stub
# ---------------------------------------------------------------------------


def test_remote_zarr_stub_raises():
    with pytest.raises(NotImplementedError):
        check_remote_zarr("s3://bucket/prefix")


def test_infer_backend_remote_zarr_assumption():
    # Non-strict (default): unknown remote scheme → assume ZARR
    assert infer_backend("s3://bucket/my-msv4") == MSv4Backend.ZARR


def test_infer_backend_remote_strict_unsupported():
    # Strict: unknown remote scheme → NotImplementedError
    with pytest.raises(NotImplementedError):
        infer_backend("s3://bucket/my-msv4", strict=True)


# ---------------------------------------------------------------------------
# Remote Zarr -> httpserver
# ---------------------------------------------------------------------------


def test_remote_zarr_v2_http(httpserver):
    httpserver.expect_request("/store/.zattrs", method="HEAD").respond_with_data("{}", status=200)
    assert check_remote_zarr(httpserver.url_for("/store")) is True


def test_remote_zarr_v3_http(httpserver):
    httpserver.expect_request("/store/zarr.json", method="HEAD").respond_with_data("{}", status=200)
    assert check_remote_zarr(httpserver.url_for("/store")) is True


def test_remote_zarr_not_found_http(httpserver):
    httpserver.expect_request("/store/.zattrs", method="HEAD").respond_with_data("", status=404)
    httpserver.expect_request("/store/zarr.json", method="HEAD").respond_with_data("", status=404)
    assert check_remote_zarr(httpserver.url_for("/store")) is False


def test_infer_backend_remote_zarr_v2(httpserver):
    """Resolves to Zarr in non-strict and Unknown in strict"""
    httpserver.expect_request("/store/.zattrs", method="HEAD").respond_with_data("", status=404)
    httpserver.expect_request("/store/zarr.json", method="HEAD").respond_with_data("", status=404)
    assert infer_backend(httpserver.url_for("/store")) == MSv4Backend.ZARR
    assert infer_backend(httpserver.url_for("/store"), strict=True) == MSv4Backend.UNKNOWN


def test_infer_backend_remote_zarr_v3(httpserver):
    """Resolves to Zarr in non-strict and Unknown in strict"""
    httpserver.expect_request("/store/.zattrs", method="HEAD").respond_with_data("", status=404)
    httpserver.expect_request("/store/zarr.json", method="HEAD").respond_with_data("", status=404)
    assert infer_backend(httpserver.url_for("/store")) == MSv4Backend.ZARR
    assert infer_backend(httpserver.url_for("/store"), strict=True) == MSv4Backend.UNKNOWN


def test_infer_backend_remote_zarr_v2_strict(httpserver):
    """Resolves to Zarr in both non-strict and strict"""
    httpserver.expect_request("/store/.zattrs", method="HEAD").respond_with_data("{}", status=200)
    assert infer_backend(httpserver.url_for("/store")) == MSv4Backend.ZARR
    assert infer_backend(httpserver.url_for("/store"), strict=True) == MSv4Backend.ZARR


def test_infer_backend_remote_zarr_v3_strict(httpserver):
    """Resolves to Zarr in both non-strict and strict"""
    httpserver.expect_request("/store/zarr.json", method="HEAD").respond_with_data("{}", status=200)
    assert infer_backend(httpserver.url_for("/store")) == MSv4Backend.ZARR
    assert infer_backend(httpserver.url_for("/store"), strict=True) == MSv4Backend.ZARR


# ---------------------------------------------------------------------------
# RDB magic -> non-http stub
# ---------------------------------------------------------------------------


def test_check_rdb_magic_stub_raises():
    # http/https is implemented; non-http schemes still raise NotImplementedError
    with pytest.raises(NotImplementedError):
        check_rdb_magic("s3://bucket/obs/1234567890.rdb")


# ---------------------------------------------------------------------------
# RDB magic — httpserver
# ---------------------------------------------------------------------------


def test_rdb_magic_matches(httpserver):
    httpserver.expect_request("/data.rdb").respond_with_data(b"REDIS0006", status=200)
    assert check_rdb_magic(httpserver.url_for("/data.rdb")) is True


def test_rdb_magic_no_match(httpserver):
    httpserver.expect_request("/data.rdb").respond_with_data(b"\x00\x00\x00\x00\x00", status=200)
    assert check_rdb_magic(httpserver.url_for("/data.rdb")) is False


# ---------------------------------------------------------------------------
# Unknown
# ---------------------------------------------------------------------------


def test_unknown_nonexistent_path():
    assert infer_backend("/nonexistent/path/to/ms") == MSv4Backend.UNKNOWN


def test_unknown_empty_string():
    assert infer_backend("") == MSv4Backend.UNKNOWN
