import pandas as pd
import os

def filter_phishing_emails(csv_path):
    """
    Filter CSV to keep only entries with label == 1 (phishing emails)
    and save to a new file.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1
    df_phishing = df[df.get("label", 0) == 1]
    
    print(f"Original dataset: {len(df)} emails")
    print(f"Phishing emails (label=1): {len(df_phishing)} emails")
    
    # Create output path
    input_dir = os.path.dirname(csv_path)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}-only-phishing.csv")
    
    # Save filtered dataset
    df_phishing.to_csv(output_path, index=False)
    print(f"Saved filtered dataset to: {output_path}")

# Run the filter
filter_phishing_emails("csv/TREC-07.csv")