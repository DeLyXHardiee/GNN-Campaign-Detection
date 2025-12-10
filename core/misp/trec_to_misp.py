import pandas as pd
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.url_extractor import extract_urls_from_text

def csv_to_misp(csv_path, misp_json_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1
    df = df[df.get("label", 0) == 1].reset_index(drop=False)
    df.rename(columns={"index": "orig_csv_index"}, inplace=True)
    
    misp_events = []
    for idx, row in df.reset_index(drop=True).iterrows():
        body = row.get("body", "")
        
        # Extract URLs from the email body
        extracted_urls = extract_urls_from_text(body) if body else []
        
        # Start with base attributes
        attributes = [
            {
                "type": "email-src",
                "value": row.get("sender", ""),
                "category": "Payload delivery"
            },
            {
                "type": "email-dst",
                "value": row.get("receiver", ""),
                "category": "Payload delivery"
            },
            {
                "type": "email-subject",
                "value": row.get("subject", ""),
                "category": "Payload delivery"
            },
            {
                "type": "email-date",
                "value": row.get("date", ""),
                "category": "Payload delivery"
            },
            {
                "type": "email-body",
                "value": body,
                "category": "Payload delivery"
            }
        ]
        
        # Add extracted URLs as separate attributes
        for url in extracted_urls:
            attributes.append({
                "type": "url",
                "value": url,
                "category": "Network activity"
            })
        
        # Also add any explicit URL from CSV if present
        csv_url = row.get("url", "")
        if csv_url and str(csv_url).strip():
            attributes.append({
                "type": "url",
                "value": csv_url,
                "category": "Network activity"
            })
        
        event = {
            "Event": {
                "info": f"TREC-07 Email {idx}",  # idx = position in filtered CSV (0..N-1)
                "email_index": int(idx),         # propagate this as the canonical email id
                "orig_csv_index": int(row.get("orig_csv_index", idx)),  # original row before filtering (for reference)
                "Attribute": attributes
            }
        }
        misp_events.append(event)
    
    # Print first 10 events
    for i, event in enumerate(misp_events[:10]):
        print(f"Event {i}:")
        print(json.dumps(event, indent=2))
        print("-" * 40)
    
    # Ensure output directory exists, then save as MISP JSON
    out_dir = os.path.dirname(misp_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(misp_json_path, "w", encoding="utf-8") as f:
        json.dump(misp_events, f, indent=2, ensure_ascii=False)
