import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_weather_dataset(start_date='2010-01-01', end_date='2023-12-31', num_samples=5000):
    """
    Generate a synthetic weather dataset with realistic variations.
    
    Parameters:
    - start_date: Start date of the dataset
    - end_date: End date of the dataset
    - num_samples: Number of data points to generate
    
    Returns:
    - pandas DataFrame with weather data
    """
    # Convert date strings to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    dates = [start + timedelta(days=x) for x in range((end - start).days + 1)]
    
    # Randomly sample dates to create desired number of samples
    sampled_dates = np.random.choice(dates, size=num_samples, replace=False)
    sampled_dates.sort()
    
    # Create base dataset
    data = pd.DataFrame({
        'date': sampled_dates,
    })
    
    # Add seasonality and random noise to temperature
    def calculate_temperature(date):
        # Base temperature model with seasonal variation
        month = date.month
        base_temp = {
            1: 0, 2: 2, 3: 7, 4: 12, 
            5: 17, 6: 22, 7: 25, 8: 24, 
            9: 20, 10: 14, 11: 8, 12: 3
        }[month]
        
        # Add random variation and daily fluctuation
        variation = np.random.normal(0, 3)  # Standard deviation of 3
        daily_variation = np.sin(date.timetuple().tm_yday * (2 * np.pi / 365)) * 5
        
        return base_temp + variation + daily_variation
    
    # Generate features with realistic correlations
    data['temperature'] = data['date'].apply(calculate_temperature)
    data['humidity'] = np.clip(
        50 + np.random.normal(0, 15) - 0.3 * data['temperature'], 
        10, 100
    )
    data['wind_speed'] = np.clip(
        np.abs(np.random.normal(10, 5) - 0.1 * data['temperature']), 
        0, 50
    )
    data['pressure'] = np.clip(
        1013 + np.random.normal(0, 10) - 0.2 * data['humidity'], 
        900, 1100
    )
    
    # Ensure data types and sort
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    return data

def save_dataset(data, filename='weather_dataset.csv'):
    """
    Save the generated dataset to a CSV file.
    
    Parameters:
    - data: pandas DataFrame with weather data
    - filename: Output filename
    """
    data.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    return filename

def main():
    # Generate and save dataset
    weather_data = generate_weather_dataset()
    save_dataset(weather_data)
    
    # Optional: Display dataset info
    print("\nDataset Summary:")
    print(weather_data.describe())

if __name__ == "__main__":
    main()