import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime
import matplotlib.pyplot as plt

class AdvancedWeatherDataGenerator:
    @staticmethod
    def generate_realistic_dataset(num_samples=10000):
        """
        Generate a comprehensive and realistic weather dataset
        """
        # Create date range
        start_date = datetime.datetime(2010, 1, 1)
        dates = [start_date + datetime.timedelta(hours=x) for x in range(num_samples)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates
        })
        
        # Sophisticated temperature modeling
        def calculate_temperature(row):
            # Base seasonal temperature
            month = row['date'].month
            day_of_year = row['date'].timetuple().tm_yday
            
            # Complex seasonal modeling
            seasonal_base = 15 + 20 * np.sin((day_of_year - 1) * (2 * np.pi / 365))
            
            # Time of day effect
            hour_effect = 5 * np.sin((row['date'].hour - 12) * (np.pi / 12))
            
            # Random variations
            random_variation = np.random.normal(0, 3)
            
            return seasonal_base + hour_effect + random_variation
        
        # Calculate temperature with advanced modeling
        df['temperature'] = df.apply(calculate_temperature, axis=1)
        
        # Correlated feature generation
        df['humidity'] = np.clip(
            70 - 0.5 * df['temperature'] + np.random.normal(0, 10), 
            10, 100
        )
        
        df['wind_speed'] = np.clip(
            10 + 0.3 * np.abs(df['temperature'] - 15) + np.random.normal(0, 3), 
            0, 50
        )
        
        df['pressure'] = np.clip(
            1013 - 0.2 * df['humidity'] + np.random.normal(0, 5), 
            980, 1030
        )
        
        return df

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def prepare_features(self, data):
        """
        Prepare features for model training
        """
        data = data.copy()
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.dayofyear
        data['hour'] = data['date'].dt.hour
        
        features = ['month', 'day_of_year', 'hour', 'humidity', 'wind_speed', 'pressure']
        X = data[features]
        y = data['temperature']
        
        return X, y
    
    def train_model(self, data):
        """
        Train Random Forest Regression model
        """
        X, y = self.prepare_features(data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        # Advanced Random Forest with more complex parameters
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            bootstrap=True
        )
        
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Model evaluation
        train_score = self.model.score(X_train_scaled, y_train_scaled)
        
        return train_score
    
    def predict_temperature(self, input_features):
        """
        Predict temperature based on input features
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Scale input features
        input_scaled = self.scaler_X.transform([input_features])
        
        # Predict and inverse transform
        prediction_scaled = self.model.predict(input_scaled)[0]
        prediction = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]
        
        return prediction

def main():
    st.title("🌦️ Advanced Weather Prediction System")
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # Dataset generation options
    num_samples = st.sidebar.slider(
        "Dataset Size", 
        min_value=5000, 
        max_value=50000, 
        value=20000
    )
    
    # Train model button
    if st.sidebar.button("Generate Dataset & Train Model"):
        # Generate dataset
        dataset = AdvancedWeatherDataGenerator.generate_realistic_dataset(num_samples)
        
        # Train model and get score
        train_score = predictor.train_model(dataset)
        
        st.sidebar.success(f"Model Trained! R² Score: {train_score:.4f}")
        
        # Store predictor in session state
        st.session_state.predictor = predictor
    
    # Prediction Interface
    st.header("Temperature Prediction")
    
    # Create columns for input
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        month = st.number_input(
            "Month", 
            min_value=1, 
            max_value=12, 
            value=datetime.datetime.now().month
        )
    
    with col2:
        day_of_year = st.number_input(
            "Day of Year", 
            min_value=1, 
            max_value=366, 
            value=datetime.datetime.now().timetuple().tm_yday
        )
    
    with col3:
        hour = st.number_input(
            "Hour of Day", 
            min_value=0, 
            max_value=23, 
            value=datetime.datetime.now().hour
        )
    
    with col4:
        humidity = st.slider(
            "Humidity (%)", 
            min_value=10, 
            max_value=100, 
            value=50
        )
    
    with col5:
        wind_speed = st.number_input(
            "Wind Speed (km/h)", 
            min_value=0.0, 
            max_value=50.0, 
            value=10.0,
            step=0.5
        )
    
    with col6:
        pressure = st.number_input(
            "Pressure (hPa)", 
            min_value=980, 
            max_value=1030, 
            value=1013
        )
    
    # Prediction Button
    if st.button("Predict Temperature"):
        try:
            # Ensure model is trained
            if 'predictor' not in st.session_state:
                st.warning("Please train the model first!")
                st.stop()
            
            # Prepare input features
            input_features = [
                month, day_of_year, hour, 
                humidity, wind_speed, pressure
            ]
            
            # Get prediction
            predictor = st.session_state.predictor
            prediction = predictor.predict_temperature(input_features)
            
            # Display prediction
            st.success(f"Predicted Temperature: {prediction:.2f}°C")
            
            # Contextual description
            if prediction < 0:
                st.info("❄️ It's freezing cold!")
            elif prediction < 10:
                st.info("🧥 Quite chilly, layer up!")
            elif prediction < 20:
                st.info("🌤️ Mild and comfortable.")
            elif prediction < 30:
                st.info("☀️ Warm and pleasant.")
            else:
                st.info("🥵 It's getting hot!")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()