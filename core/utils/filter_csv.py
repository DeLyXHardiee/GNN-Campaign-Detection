import pandas as pd
import os

def filter_months(df, months=6):
    """
    Finds the consecutive range of months (of length `months`) with the most emails 
    and filters out all other emails outside this range.
    """
    if not months:
        return df
        
    print(f"Filtering for best consecutive {months} months...")
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert date column to datetime
    # errors='coerce' will turn unparseable dates into NaT
    df['temp_ts'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Drop rows where date could not be parsed
    df = df.dropna(subset=['temp_ts'])
    
    # Extract Year-Month
    df['month_year'] = df['temp_ts'].dt.to_period('M')
    
    # Count emails per month and sort chronologically
    monthly_counts = df['month_year'].value_counts().sort_index()
    
    if monthly_counts.empty:
        return df

    # Reindex to fill gaps in the timeline with 0
    full_range = pd.period_range(start=monthly_counts.index.min(), end=monthly_counts.index.max(), freq='M')
    
    if len(full_range) <= months:
         print(f"Dataset span ({len(full_range)} months) is <= requested window ({months} months). Returning all.")
         df = df.drop(columns=['temp_ts', 'month_year'])
         return df
         
    monthly_counts = monthly_counts.reindex(full_range, fill_value=0)

    # Find best window using rolling sum
    rolling_sums = monthly_counts.rolling(window=months).sum()
    
    # The index of the max value corresponds to the END of the window
    best_end_period = rolling_sums.idxmax()
    best_start_period = best_end_period - (months - 1)
    
    print(f"Selected time range: {best_start_period} to {best_end_period} (Total emails in range: {int(rolling_sums[best_end_period])})")
    
    # Filter
    mask = (df['month_year'] >= best_start_period) & (df['month_year'] <= best_end_period)
    df_filtered = df[mask].copy()
    
    # Cleanup
    df_filtered = df_filtered.drop(columns=['temp_ts', 'month_year'])
    
    return df_filtered

def filter_phishing_emails(csv_path, months=6):
    """
    Filter CSV to keep only entries with label == 1 (phishing emails)
    and save to a new file. Optionally filter by top N months.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1
    df_phishing = df[df.get("label", 0) == 1]
    
    print(f"Original dataset: {len(df)} emails")
    print(f"Phishing emails (label=1): {len(df_phishing)} emails")
    
    if months:
        df_phishing = filter_months(df_phishing, months)
        print(f"Phishing emails after month filtering: {len(df_phishing)} emails")
    
    # Create output path
    input_dir = os.path.dirname(csv_path)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}-only-phishing-{months}m.csv")
    
    # Save filtered dataset
    df_phishing.to_csv(output_path, index=False)
    print(f"Saved filtered dataset to: {output_path}")

# Run the filter
filter_phishing_emails("../data/csv/TREC-07.csv")