"""
URL extraction and parsing utilities for email campaign detection.

Provides functions to:
- Extract URLs from text using regex
- Parse URLs into components (domain, stem/path)
- Normalize URLs for deduplication
"""
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse


URL_PATTERN = re.compile(
    r'(?:https?://|www\.)[^\s\'"<>]+',
    re.IGNORECASE
)


def extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    
    urls = URL_PATTERN.findall(text)
    cleaned = []
    for url in urls:
        # Remove trailing punctuation
        url = url.rstrip('.,;:!?)')
        if url:
            cleaned.append(url)
    
    return cleaned


def parse_url_components(url: str) -> Dict[str, Any]:
    if not url:
        return {
            "full_url": "",
            "domain": "",
            "stem": "",
            "scheme": "",
        }
    
    try:
        parsing_url = url
        if not url.lower().startswith(('http://', 'https://')):
            parsing_url = 'http://' + url
            
        parsed = urlparse(parsing_url)
        
        domain = parsed.netloc.lower() if parsed.netloc else ""
        
        stem_parts = []
        if parsed.path:
            stem_parts.append(parsed.path)
        if parsed.params:
            stem_parts.append(f";{parsed.params}")
        if parsed.query:
            stem_parts.append(f"?{parsed.query}")
        if parsed.fragment:
            stem_parts.append(f"#{parsed.fragment}")
        
        stem = "".join(stem_parts) if stem_parts else "/"
        
        normalized = urlunparse((
            parsed.scheme.lower(),
            domain,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        return {
            "full_url": normalized,
            "domain": domain,
            "stem": stem,
            "scheme": parsed.scheme.lower() if parsed.scheme else "",
        }
    except Exception:
        return {
            "full_url": url,
            "domain": "",
            "stem": "",
            "scheme": "",
        }


def extract_and_parse_urls(text: str) -> List[Dict[str, Any]]:
    urls = extract_urls_from_text(text)
    return [parse_url_components(url) for url in urls]


def deduplicate_urls(url_list: List[str]) -> List[str]:
    seen = set()
    result = []
    for url in url_list:
        normalized = url.lower().rstrip('/')
        if normalized not in seen:
            seen.add(normalized)
            result.append(url)
    return result
