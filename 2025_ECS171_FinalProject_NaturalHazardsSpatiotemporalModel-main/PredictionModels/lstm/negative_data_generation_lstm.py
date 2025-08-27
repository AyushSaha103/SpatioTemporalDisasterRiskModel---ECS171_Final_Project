from scipy.spatial import cKDTree
import pandas as pd
import numpy as np

# df = pd.read_csv('./datasets/DisasterDeclarationSummariesPlusLocation_ArtificiallyClustered.csv')
# display(df)

df = pd.read_csv('datasets/UpdatedDisasterDeclarationSummariesPlusLocation_RowColFiltered.csv')
df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
df['disasterOccurred'] = 1
# Define a minimum distance threshold (in degrees) to avoid placing "no disaster" points too close to real disasters
min_distance_threshold = 10.0

# Extract real disaster locations and times
real_disaster_points = df[['latitude', 'longitude', 'timeOfStartInDays']].values

# Build a KDTree for efficient nearest-neighbor search
tree = cKDTree(real_disaster_points)

# Generate synthetic "no disaster" points
num_synthetic_points = len(df)  # Generate the same number of synthetic points as real disasters
synthetic_no_disaster_points = []

while len(synthetic_no_disaster_points) < num_synthetic_points:
    # Randomly sample latitude, longitude, and time within data range
    lat = np.random.uniform(df['latitude'].min(), df['latitude'].max())
    lon = np.random.uniform(df['longitude'].min(), df['longitude'].max())
    time = np.random.uniform(df['timeOfStartInDays'].min(), df['timeOfStartInDays'].max())

    # Check the minimum distance condition
    dist, _ = tree.query([lat, lon, time], k=1)
    if dist > min_distance_threshold:
        synthetic_no_disaster_points.append((lat, lon, time))  # -1 as a placeholder for "no disaster"

# Convert to DataFrame
df_no_disaster = pd.DataFrame(synthetic_no_disaster_points, columns=['latitude', 'longitude', 'timeOfStartInDays'])
df_no_disaster["disasterOccurred"] = 0  # Mark these as non-disasters

# Combine real disaster data with synthetic non-disaster data
df_balanced = pd.concat([df, df_no_disaster], ignore_index=True)

# Save the updated dataset
balanced_file_path_updated = "datasets/DisasterDataset_WithBetterNoDisasters.csv"
df_balanced.to_csv(balanced_file_path_updated, index=False)

balanced_file_path_updated

print(df_balanced)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# Assuming df_balanced is your dataframe with the disaster data

# Select half of the data for plotting
half_df = df_balanced.sample(frac=0.5, random_state=42)  # Select half of the data randomly

# Extract the necessary columns
longitude = half_df['longitude']
latitude = half_df['latitude']
time_in_days = half_df['timeOfStartInDays']
disaster_occurred = half_df['disasterOccurred']

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color map
sc = ax.scatter(longitude, latitude, time_in_days, c=disaster_occurred, cmap='coolwarm', marker='o')

# Add color bar to represent disasterOccurred values
plt.colorbar(sc, label='Disaster Occurred (1=Yes, 0=No)')

# Set axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Time in Days')

# Set the title
ax.set_title('Disasters and Non-Disasters in 3D (Half Data)')

# Show the plot
plt.show()

