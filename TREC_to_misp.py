import pandas as pd
import json

def csv_to_misp(csv_path, misp_json_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1
    df = df[df.get("label", 0) == 1]
    
    misp_events = []
    for idx, row in df.iterrows():
        event = {
            "Event": {
                "info": f"TREC-07 Email {idx}",
                "Attribute": [
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
                        "value": row.get("body", ""),
                        "category": "Payload delivery"
                    },
                    {
                        "type": "url",
                        "value": row.get("url", ""),
                        "category": "Network activity"
                    }
                ]
            }
        }
        misp_events.append(event)
    
    # Print first 10 events
    for i, event in enumerate(misp_events[:10]):
        print(f"Event {i+1}:")
        print(json.dumps(event, indent=2))
        print("-" * 40)
    
    # Save as MISP JSON
    with open(misp_json_path, "w") as f:
        json.dump(misp_events, f, indent=2)

if __name__ == "__main__":
    csv_to_misp("TREC-07.csv", "trec07_misp.json")