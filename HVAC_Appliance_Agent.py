import datetime
from typing import Dict, List
import numpy as np
from sklearn.ensemble import IsolationForest

class HVACMonitoringAgent:
    def __init__(self, building_type: str):
        self.building_type = building_type
        self.sensor_data = {}
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.energy_consumption_history = []

    def process_sensor_data(self, data: Dict[str, float]):
        """Process incoming sensor data"""
        timestamp = datetime.datetime.now()
        self.sensor_data[timestamp] = data
        self.detect_anomalies()
        self.optimize_energy_consumption()

    def detect_anomalies(self):
        """Detect anomalies in sensor data"""
        recent_data = list(self.sensor_data.values())[-100:]  # Last 100 data points
        if len(recent_data) == 100:
            X = np.array(recent_data)
            anomaly_results = self.anomaly_detector.fit_predict(X)
            if -1 in anomaly_results:
                print("Anomaly detected in recent sensor data!")

    def optimize_energy_consumption(self):
        """Optimize energy consumption based on current conditions and building type"""
        current_temp = self.sensor_data[max(self.sensor_data.keys())]['temperature']
        current_humidity = self.sensor_data[max(self.sensor_data.keys())]['humidity']

        if self.building_type == 'hospital':
            optimal_temp = 22  # Example: hospitals might need stricter temperature control
        elif self.building_type == 'office':
            optimal_temp = 23  # Example: offices might allow more flexibility
        else:  # home
            optimal_temp = 24  # Example: homes might prioritize energy savings

        if current_temp > optimal_temp + 1:
            print(f"Lowering temperature to {optimal_temp}°C")
        elif current_temp < optimal_temp - 1:
            print(f"Raising temperature to {optimal_temp}°C")

        # Humidity control
        if current_humidity > 60:
            print("Activating dehumidifier")
        elif current_humidity < 30:
            print("Activating humidifier")

    def generate_report(self) -> str:
        """Generate a report of system status and recommendations"""
        latest_data = self.sensor_data[max(self.sensor_data.keys())]
        report = f"Building Type: {self.building_type}\n"
        report += f"Current Temperature: {latest_data['temperature']}°C\n"
        report += f"Current Humidity: {latest_data['humidity']}%\n"
        report += f"Energy Consumption: {latest_data['energy_consumption']} kWh\n"
        
        # Add more detailed analysis and recommendations here
        
        return report

# Example usage
agent = HVACMonitoringAgent('hospital')

# Simulate incoming sensor data
for _ in range(100):
    fake_data = {
        'temperature': np.random.normal(22, 2),
        'humidity': np.random.normal(50, 10),
        'energy_consumption': np.random.normal(100, 20)
    }
    agent.process_sensor_data(fake_data)

print(agent.generate_report())
