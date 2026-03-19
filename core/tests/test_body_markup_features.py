from core.feature_set_extraction.feature_set_extraction import extract_body_based_features


def test_body_features_include_binary_markup_presence_and_image_count_from_structured_data():
    body = "Plain body"
    html = {
        "tag_counts": {"html": 1, "body": 1, "img": 2, "script": 1, "style": 1, "a": 3},
        "tree_stats": {"images": 2, "external_scripts": 1},
        "structure_fingerprint": "abc",
    }
    css = {
        "style_features": {
            "unique_color_count": 3,
            "primary_color": "#ffffff",
            "uses_media_queries": True,
        }
    }

    features = extract_body_based_features(body, html=html, css=css)

    assert features["has_html_tags"] == 1
    assert features["has_images"] == 1
    assert features["num_images"] == 2
    assert features["has_script"] == 1
    assert features["has_css_specs"] == 1
    assert isinstance(features["bow"], dict)
    assert features["bow"].get("plain", 0) == 1
    assert features["bow"].get("body", 0) == 1


def test_body_features_include_binary_markup_presence_and_image_count_from_html_body():
    body = "<html><body><style>.x{color:red;}</style><script>alert(1)</script><img src='a.png'/>Hi</body></html>"
    features = extract_body_based_features(body, html={}, css={})

    assert features["has_html_tags"] == 1
    assert features["has_images"] == 1
    assert features["num_images"] == 1
    assert features["has_script"] == 1
    assert features["has_css_specs"] == 1
    assert isinstance(features["bow"], dict)
    assert features["bow"].get("html", 0) >= 1
