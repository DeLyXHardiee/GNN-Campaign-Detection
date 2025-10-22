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


# Regex pattern for matching http/https URLs
URL_PATTERN = re.compile(
    r'https?://[^\s\'"<>]+',
    re.IGNORECASE
)


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract all HTTP(S) URLs from text using regex.
    
    Args:
        text: Input text (e.g., email body)
    
    Returns:
        List of URL strings found in the text
    """
    if not text:
        return []
    
    urls = URL_PATTERN.findall(text)
    # Clean up common trailing punctuation that gets captured
    cleaned = []
    for url in urls:
        # Remove trailing punctuation
        url = url.rstrip('.,;:!?)')
        if url:
            cleaned.append(url)
    
    return cleaned


def parse_url_components(url: str) -> Dict[str, Any]:
    """
    Parse a URL into its components.
    
    Returns a dict with:
    - full_url: normalized full URL
    - domain: domain name (e.g., "example.com")
    - stem: path + params + query (e.g., "/path?query=1")
    - scheme: http or https
    """
    if not url:
        return {
            "full_url": "",
            "domain": "",
            "stem": "",
            "scheme": "",
        }
    
    try:
        parsed = urlparse(url)
        
        # Extract domain (netloc)
        domain = parsed.netloc.lower() if parsed.netloc else ""
        
        # Extract stem: path + params + query + fragment
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
        
        # Reconstruct normalized URL
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
        # If parsing fails, return the original URL with empty components
        return {
            "full_url": url,
            "domain": "",
            "stem": "",
            "scheme": "",
        }


def extract_and_parse_urls(text: str) -> List[Dict[str, Any]]:
    """
    Extract URLs from text and parse each into components.
    
    Args:
        text: Input text
    
    Returns:
        List of dicts, each containing parsed URL components
    """
    urls = extract_urls_from_text(text)
    return [parse_url_components(url) for url in urls]


def deduplicate_urls(url_list: List[str]) -> List[str]:
    """
    Remove duplicate URLs while preserving order.
    
    Args:
        url_list: List of URL strings
    
    Returns:
        Deduplicated list of URLs
    """
    seen = set()
    result = []
    for url in url_list:
        normalized = url.lower().rstrip('/')
        if normalized not in seen:
            seen.add(normalized)
            result.append(url)
    return result
