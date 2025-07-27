import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_devices = 500

# Generate device IDs
device_ids = [f"D{i+1}" for i in range(n_devices)]

# Simulate memory, traffic, and latency with normal distribution
memory = np.random.normal(loc=60, scale=15, size=n_devices).clip(0, 100)
traffic = np.random.normal(loc=40, scale=10, size=n_devices).clip(0, 100)
latency = np.random.normal(loc=30, scale=20, size=n_devices).clip(0, 100)

# Calculate fault_rate with weighted formula
a, b, c = 0.5, 0.3, 0.2
fault_rate = a * memory + b * traffic + c * latency

# Inject random noise to simulate real-world variance
fault_rate += np.random.normal(loc=0, scale=5, size=n_devices)

# Label as 'faulty' if fault_rate > 50 (approx 40% devices)
threshold = 50
faulty = (fault_rate > threshold).astype(int)

# Create dataframe
df = pd.DataFrame({
    "device": device_ids,
    "memory": memory,
    "traffic": traffic,
    "latency": latency,
    "fault_rate": fault_rate,
    "faulty": faulty
})

# Save to CSV
df.to_csv("device_data.csv", index=False)
print("✅ Realistic device_data.csv generated with 500 samples.")
