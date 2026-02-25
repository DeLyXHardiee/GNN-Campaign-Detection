from core.preprocessing.data_parser import _extract_selected_headers


def test_received_filters_internal_relays_and_duplicates():
    headers = {
        "Received": (
            "from nam12-prod.outlook.com by relay.outlook.com with ESMTP; Tue | "
            "from smtp01.corp.local by edge.local with ESMTP; Tue | "
            "from public.mx.example.com by inbound.company.net with ESMTP id abc123; Tue | "
            "from public.mx.example.com by inbound.company.net with ESMTP id abc123; Tue"
        )
    }

    selected = _extract_selected_headers(headers)
    assert selected["Received"] == [
        {
            "origin_ip": "",
            "helo_host": "nam12-prod.outlook.com",
            "by_host": "relay.outlook.com",
            "timestamp": "Tue",
        },
        {
            "origin_ip": "",
            "helo_host": "public.mx.example.com",
            "by_host": "inbound.company.net",
            "timestamp": "Tue",
        },
    ]


def test_return_path_normalization_and_domain_extraction():
    headers = {"Return-Path": "<Sender+Tag@Example.COM>"}
    selected = _extract_selected_headers(headers)
    assert selected["Return-Path"] == {"email": "sender+tag@example.com", "domain": "example.com"}


def test_content_type_removes_boundary_and_charset():
    headers = {
        "Content-Type": 'multipart/alternative; boundary="----=_Part_12345_67890"; charset=UTF-8; format=flowed'
    }
    selected = _extract_selected_headers(headers)
    assert selected["Content-Type"] == "multipart/alternative; format=flowed"


def test_received_spf_keeps_compact_values_only():
    headers = {
        "Received-SPF": (
            "Pass (protection.outlook.com: domain of sender.example.com designates 192.0.2.1 as permitted sender) "
            "receiver=mail.contoso.com; client-ip=192.0.2.1; helo=mx.sender.example.com; "
            "envelope-from=bounce@sender.example.com"
        )
    }
    selected = _extract_selected_headers(headers)
    assert selected["Received-SPF"] == (
        "spf=pass; domain=sender.example.com; client-ip=192.0.2.1; "
        "helo=mx.sender.example.com; envelope-from=bounce@sender.example.com"
    )


def test_list_unsubscribe_strips_tracking_and_tokens_but_keeps_base_url():
    headers = {
        "List-Unsubscribe": (
            "<https://news.example.com/unsubscribe/aaaaaaaaaaaaaaaaaaaa?utm_source=newsletter&token=secret&uid=12345>, "
            "<mailto:leave@example.com?subject=unsubscribe&user_id=aaaaaaaaaaaaaaaaaaaa>"
        )
    }
    selected = _extract_selected_headers(headers)
    assert selected["List-Unsubscribe"] == (
        "https://news.example.com/unsubscribe | mailto:leave@example.com?subject=unsubscribe"
    )


def test_authentication_results_keeps_compact_auth_outcomes():
    headers = {
        "Authentication-Results": (
            "mx.example.net; spf=pass smtp.mailfrom=sender.example.com; "
            "dkim=pass header.i=@sender.example.com; dmarc=pass action=none header.from=sender.example.com; "
            "arc=pass (i=1); arc-seal=none; reason=250 2.0.0"
        )
    }
    selected = _extract_selected_headers(headers)
    assert selected["Authentication-Results"] == "spf=pass; dkim=pass; dmarc=pass; header.from=sender.example.com"


def test_extract_selected_headers_integration_for_all_new_filters():
    headers = {
        "Received": "from internal.prod.outlook.com by mx; Tue | from public.mail.net by mx.target.net with ESMTP; Tue",
        "Return-Path": "<BOUNCE@Example.Org>",
        "Content-Type": "text/plain; charset=UTF-8; boundary=abc123xyz; format=flowed",
        "Received-SPF": "SoftFail (domain of bad.example does not designate 198.51.100.77) client-ip=198.51.100.77",
        "List-Unsubscribe": "<https://example.org/unsub?gclid=abc123&topic=alerts>",
        "Authentication-Results": "mx; dkim=fail reason=bad sig; spf=neutral; dmarc=fail header.from=example.org; arc=pass",
    }

    selected = _extract_selected_headers(headers)
    assert selected["Received"] == [
        {
            "origin_ip": "",
            "helo_host": "internal.prod.outlook.com",
            "by_host": "mx",
            "timestamp": "Tue",
        },
        {
            "origin_ip": "",
            "helo_host": "public.mail.net",
            "by_host": "mx.target.net",
            "timestamp": "Tue",
        },
    ]
    assert selected["Return-Path"] == {"email": "bounce@example.org", "domain": "example.org"}
    assert selected["Content-Type"] == "text/plain; format=flowed"
    assert selected["Received-SPF"] == "spf=softfail; domain=bad.example"
    assert selected["List-Unsubscribe"] == "https://example.org/unsub?topic=alerts"
    assert selected["Authentication-Results"] == "spf=neutral; dkim=fail; dmarc=fail; header.from=example.org"
