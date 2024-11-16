import eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set your LSEG (Eikon) API key
ek.set_app_key('abde57015b854677872e25ff39a04988bf949d92')

def fetch_berkshire_data():
    """
    Fetch 10 years of Berkshire Hathaway stock data
    Using BRKa.N for Berkshire Hathaway Class A shares
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # Approximately 10 years
    
    try:
        # Fetch time series data
        df = ek.get_timeseries(
            "BRKa.N",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='daily'
        )
        
        # Rename columns to match standard format
        df = df.rename(columns={
            'HIGH': 'High',
            'LOW': 'Low',
            'OPEN': 'Open',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def clean_and_process_data(df):
    """
    Clean and process the raw data, calculate additional metrics
    """
    if df is None:
        return None
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Ensure all numeric columns are float
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Calculate Adjusted Close (simplified - you may need to adjust based on actual split/dividend data)
    df['Adj_Close'] = df['Close']
    
    # Calculate log returns
    df['Return'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))
    
    # Remove the first row since it will have NaN return
    df = df.dropna()
    
    # Ensure data is sorted by date
    df = df.sort_values('Date')
    
    # Reset index after all processing
    df = df.reset_index(drop=True)
    
    return df

def calculate_volatility_metrics(df):
    """
    Calculate additional volatility metrics useful for ARCH/GARCH modeling
    """
    if df is None:
        return None
    
    # Calculate squared returns (for ARCH/GARCH modeling)
    df['Return_Squared'] = df['Return'] ** 2
    
    # Calculate rolling volatility (20-day window)
    df['Rolling_Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Calculate absolute returns
    df['Abs_Return'] = np.abs(df['Return'])
    
    return df

def prepare_arch_garch_data():
    """
    Main function to prepare data for ARCH/GARCH modeling
    """
    # Fetch raw data
    raw_data = fetch_berkshire_data()
    
    # Clean and process the data
    processed_data = clean_and_process_data(raw_data)
    
    # Calculate volatility metrics
    final_data = calculate_volatility_metrics(processed_data)
    
    if final_data is not None:
        # Select and reorder columns for final output
        columns = [
            'Date', 'Open', 'High', 'Low', 'Close', 
            'Volume', 'Adj_Close', 'Return', 'Return_Squared',
            'Rolling_Volatility', 'Abs_Return'
        ]
        final_data = final_data[columns]
        
        # Display basic statistics
        print("\nData Summary:")
        print(final_data.describe())
        
        # Check for any remaining missing values
        print("\nMissing Values:")
        print(final_data.isnull().sum())
        
        return final_data
    
    return None

# Execute the data preparation
if __name__ == "__main__":
    data = prepare_arch_garch_data()
    
    if data is not None:
        # Save to CSV for further analysis
        data.to_csv('berkshire_data_for_arch_garch.csv', index=False)
        print("\nData has been saved to 'berkshire_data_for_arch_garch.csv'")
