from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

ExtractorStrategy = Literal["email_list", "url_list", "string_single", "dict_mapping", "string_list"]


@dataclass(frozen=True)
class AttributeMapping:
    field: str
    strategy: ExtractorStrategy
    accumulate: bool = False
    extract_urls_side_effect: bool = False
    lowercase_items: bool = False


@dataclass(frozen=True)
class AttributeSchema:
    exact: Dict[str, AttributeMapping]
    contains: List[Tuple[str, AttributeMapping]]

    def resolve(self, attr_type: str) -> AttributeMapping | None:
        key = (attr_type or "").strip().lower()
        if key in self.exact:
            return self.exact[key]
        for needle, mapping in self.contains:
            if needle in key:
                return mapping
        return None


DEFAULT_MISP_ATTRIBUTE_SCHEMA = AttributeSchema(
    exact={
        "email-src": AttributeMapping("senders", "email_list", accumulate=True),
        "from": AttributeMapping("senders", "email_list", accumulate=True),
        "email-dst": AttributeMapping("receivers", "email_list", accumulate=True),
        "email-cc": AttributeMapping("receivers", "email_list", accumulate=True),
        "email-bcc": AttributeMapping("receivers", "email_list", accumulate=True),
        "to": AttributeMapping("receivers", "email_list", accumulate=True),
        "email-subject": AttributeMapping("subject", "string_single"),
        "subject": AttributeMapping("subject", "string_single"),
        "email-body": AttributeMapping("body", "string_single", extract_urls_side_effect=True),
        "body": AttributeMapping("body", "string_single", extract_urls_side_effect=True),
        "html": AttributeMapping("html", "dict_mapping", extract_urls_side_effect=True),
        "css": AttributeMapping("css", "dict_mapping", extract_urls_side_effect=True),
        "url": AttributeMapping("urls", "url_list", accumulate=True),
        "email-date": AttributeMapping("date", "string_single"),
        "date": AttributeMapping("date", "string_single"),
        "header_list-unsubscribe": AttributeMapping("urls", "url_list", accumulate=True),
    },
    contains=[
        ("attachment", AttributeMapping("attachments", "string_list", accumulate=True, lowercase_items=True)),
    ],
)


__all__ = ["AttributeMapping", "AttributeSchema", "DEFAULT_MISP_ATTRIBUTE_SCHEMA", "ExtractorStrategy"]
