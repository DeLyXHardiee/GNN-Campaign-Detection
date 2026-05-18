"""Tests for HTML href + linkify-it URL extraction."""

from core.preprocessing.utils.url_extractor import (
    extract_hrefs_from_html,
    extract_urls_from_plain_and_html,
    extract_urls_from_text,
    normalize_http_url,
    parse_url_components,
    refang_url_like_schemes,
)


def test_refang_restores_schemes():
    assert refang_url_like_schemes("see hxxps://evil.com/x") == "see https://evil.com/x"


def test_td_padding_not_treated_as_url():
    assert normalize_http_url("td.padding") is None
    assert normalize_http_url("https://td.padding/x") is None


def test_mailto_not_normalized_as_https():
    assert normalize_http_url("mailto:user@example.com") is None


def test_real_domain_with_suffix_accepted():
    assert normalize_http_url("https://example.com/p") == "https://example.com/p"


def test_normalize_rejects_javascript():
    assert normalize_http_url("javascript:alert(1)") is None


def test_normalize_rejects_malformed_bracketed_host():
    assert normalize_http_url("http://[.]/path") is None
    assert normalize_http_url("[.]") is None


def test_extract_urls_from_defanged_plain_text():
    text = "open hxxps://phish.evil.com/path?q=1 for details"
    urls = extract_urls_from_text(text)
    assert urls == ["https://phish.evil.com/path?q=1"]


def test_linkify_fuzzy_www():
    text = "visit www.example.org/foo ok"
    urls = extract_urls_from_text(text)
    assert len(urls) == 1
    assert "example.org" in urls[0]


def test_html_hrefs_and_base():
    html = """
    <html><head><base href="https://cdn.example.com/root/"/></head>
    <body>
      <a href="/rel">x</a>
      <a href="https://abs.other.org/page">y</a>
      <area href="hxxps://defangedevil.com/t" alt="z"/>
    </body></html>
    """
    hrefs = extract_hrefs_from_html(html)
    assert "https://cdn.example.com/rel" in hrefs
    assert "https://abs.other.org/page" in hrefs
    assert "https://defangedevil.com/t" in hrefs


def test_javascript_href_filtered_from_combined():
    html = '<a href="javascript:void(0)">x</a><a href="https://ok.example.com/">y</a>'
    urls = extract_urls_from_plain_and_html("", html)
    assert urls == ["https://ok.example.com/"]


def test_dedupe_plain_vs_html():
    plain = "see https://dup.example.com/a"
    html = '<a href="https://dup.example.com/a">same</a>'
    urls = extract_urls_from_plain_and_html(plain, html)
    assert urls == ["https://dup.example.com/a"]


def test_parse_url_components_defanged_https_regression():
    """Must not prepend http:// to defanged URLs (would yield domain ``hxxps:``)."""
    for raw in ("hxxps://lkos.de/", "hxxps://pfa.dk/"):
        comp = parse_url_components(raw)
        assert comp["domain"] == raw.split("//")[1].split("/")[0].lower(), raw
        assert comp["domain"] != "hxxps"
        assert "hxxps:" not in comp["domain"]
        assert comp["stem"] == "/", raw
        assert comp["full_url"].lower().startswith("https://")


def test_parse_url_components_defanged_http_and_https_paths():
    c_http = parse_url_components("hxxp://example.com/path")
    assert c_http["domain"] == "example.com"
    assert c_http["scheme"] == "http"
    assert c_http["stem"] == "/path"
    assert c_http["full_url"].startswith("http://example.com/path")

    c_https = parse_url_components("hxxps://example.com/path?x=1")
    assert c_https["domain"] == "example.com"
    assert c_https["scheme"] == "https"
    assert "/path" in c_https["stem"]
    assert c_https["full_url"].startswith("https://example.com/")


def test_parse_url_components_defanged_case_insensitive():
    comp = parse_url_components("HXXPS://Example.COM/foo")
    assert comp["domain"] == "example.com"
    assert comp["full_url"].lower().startswith("https://example.com")


def test_parse_url_host_defanged_yields_registrable_domain():
    from core.feature_set_extraction.url_extraction_utils import (
        parse_url_host_and_registrable_domain,
    )

    host, reg, ok = parse_url_host_and_registrable_domain("hxxps://lkos.de/path")
    assert ok
    assert host == "lkos.de"
    assert reg == "lkos.de"
