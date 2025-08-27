import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib 

MODEL_PATH = "disaster_lstm.pth"
SCALER_PATH = "scaler.pkl"

class DisasterLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(DisasterLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # First layer to extract spatial features
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, num_layers=3)
        self.fc2 = nn.Linear(64, output_size)  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # Transform input before LSTM
        x = self.relu(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        _, (h_n, _) = self.lstm(x)
        x = self.fc2(h_n[-1]) 
        x = self.sigmoid(x)
        return x

model = DisasterLSTM(3, 1)
model.load_state_dict(torch.load('disaster_lstm.pth'))
model.eval()

scaler = joblib.load(SCALER_PATH)

# Load the dataset
file_path = "../../Datasets/DisasterDataset_WithBetterNoDisasters.csv"
df = pd.read_csv(file_path)

# Ensure 'timeOfStartInDays' column exists and filter the relevant time range
time_range = (20000,20365)

df_filtered = df[(df["timeOfStartInDays"] >= time_range[0]) & (df["timeOfStartInDays"] <= time_range[1])]
# Check available disaster type columns
disaster_columns = [col for col in df_filtered.columns if "disasterType_" in col]

# Assign a color for each disaster type
colors = plt.cm.get_cmap("tab10", len(disaster_columns)+1)
total = 0
# Plot the disasters on a scatter plot
plt.figure(figsize=(10, 6))
for i, disaster in enumerate(disaster_columns):
    subset = df_filtered[df_filtered[disaster] == 1]  # Get rows where this disaster type occurred
    # print(df_filtered[disaster])
    plt.scatter(subset["longitude"], subset["latitude"], label=disaster, alpha=0.6, color=colors(i))
    total += len(subset)
no_disasters = df_filtered[df_filtered['disasterOccurred'] == 0]
plt.scatter(no_disasters['longitude'], no_disasters['latitude'], label='no disaster', alpha=0.6, color=colors(len(disaster_columns)))
print(total)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Disasters Between timeOfStartInDays {time_range[0]} and {time_range[1]}")

plt.legend(loc="lower right", fontsize="small")
plt.grid(True)
plt.show()



lats = df_filtered[['latitude']]
lons = df_filtered[['longitude']]
time_fixed = df_filtered[['timeOfStartInDays']]
data_points = np.column_stack((lats, lons, time_fixed))

# Normalize data (Ensure you use the same scaler as training)
data_points_scaled = scaler.fit_transform(data_points)  # Fit/transform for demo (use pre-trained scaler)

# Convert to Tensor for Model Input
input_tensor = torch.tensor(data_points_scaled, dtype=torch.float32) # Shape (batch, seq, features)

# Make Predictions
with torch.no_grad():
    predictions = model(input_tensor).numpy().flatten()  # Get probabilities

# Plot Results - Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(lons, lats, c=predictions, cmap="inferno", alpha=0.5)
plt.colorbar(label="Predicted Probability")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted Event Probability Across USA")
plt.show()

# Convert to Heatmap Grid
num_grid = 100  # Grid resolution
lat_grid = np.linspace(lat_min, lat_max, num_grid)
lon_grid = np.linspace(lon_min, lon_max, num_grid)
grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
