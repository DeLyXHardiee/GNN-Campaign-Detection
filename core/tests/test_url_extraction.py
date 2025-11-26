"""
Quick test/demo of URL extraction functionality.

Run this to verify URL extraction is working correctly.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.utils.url_extractor import extract_urls_from_text, parse_url_components, extract_and_parse_urls


def test_url_extraction():
    """Test URL extraction from sample text."""
    
    sample_text = """
    Click here to visit http://example.com/path?query=test for great deals!
    Also check out https://spam-site.net/offers and https://another-site.org/page.
    Visit www.example.com (not a valid URL for our regex).
    """
    
    print("=" * 60)
    print("URL EXTRACTION TEST")
    print("=" * 60)
    print(f"\nSample text:\n{sample_text}\n")
    
    # Extract URLs
    urls = extract_urls_from_text(sample_text)
    print(f"Extracted {len(urls)} URLs:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
    
    print("\n" + "=" * 60)
    print("URL COMPONENT PARSING")
    print("=" * 60)
    
    # Parse each URL
    for url in urls:
        parsed = parse_url_components(url)
        print(f"\nURL: {url}")
        print(f"  Domain: {parsed['domain']}")
        print(f"  Stem:   {parsed['stem']}")
        print(f"  Scheme: {parsed['scheme']}")
    
    print("\n" + "=" * 60)
    print("DEDUPLICATION TEST")
    print("=" * 60)
    
    # Test deduplication
    duplicate_text = """
    Visit http://example.com and http://example.com again.
    Also http://EXAMPLE.COM (case difference).
    And http://example.com/ (trailing slash).
    """
    
    print(f"\nText with duplicates:\n{duplicate_text}")
    urls = extract_urls_from_text(duplicate_text)
    print(f"\nExtracted {len(urls)} URLs (with duplicates):")
    for url in urls:
        print(f"  - {url}")
    
    from core.utils.url_extractor import deduplicate_urls
    unique = deduplicate_urls(urls)
    print(f"\nAfter deduplication: {len(unique)} unique URLs:")
    for url in unique:
        print(f"  - {url}")
    
    print("\n" + "=" * 60)
    print("DOMAIN/STEM SHARING TEST")
    print("=" * 60)
    
    sharing_text = """
    http://example.com/page1
    http://example.com/page2
    http://different.com/page1
    """
    
    print(f"\nText showing domain/stem sharing:\n{sharing_text}")
    parsed_list = extract_and_parse_urls(sharing_text)
    
    domains = {}
    stems = {}
    
    for parsed in parsed_list:
        url = parsed['full_url']
        domain = parsed['domain']
        stem = parsed['stem']
        
        domains.setdefault(domain, []).append(url)
        stems.setdefault(stem, []).append(url)
    
    print("\nDomain grouping (URLs sharing the same domain):")
    for domain, url_list in domains.items():
        print(f"  Domain '{domain}':")
        for url in url_list:
            print(f"    - {url}")
    
    print("\nStem grouping (URLs sharing the same path):")
    for stem, url_list in stems.items():
        print(f"  Stem '{stem}':")
        for url in url_list:
            print(f"    - {url}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_url_extraction()
