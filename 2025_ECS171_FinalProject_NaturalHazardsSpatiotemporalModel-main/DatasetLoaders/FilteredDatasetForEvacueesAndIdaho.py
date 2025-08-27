import pandas as pd
import os

folder_path = '/Users/veronica/Desktop/2025_ECS171_FinalProject_NaturalHazardsSpatiotemporalModel/datasets/'

df = pd.read_csv('/Users/veronica/Desktop/2025_ECS171_FinalProject_NaturalHazardsSpatiotemporalModel/PredictionModels/DisasterDeclarationSummariesPlusLocation_RowColFiltered.csv')

df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'], errors='coerce')

# Longitude and latitude bounds for Idaho
min_lat = 41.9880
max_lat = 49.0030
min_lon = -117.2437
max_lon = -111.043

# Filter out values of 'HURRICANE KATRINA EVACUATION' or 'HURRICANE KATRINA EVACUEES'
df_filtered = df[~df['declarationTitle'].str.contains('HURRICANE KATRINA EVACUATION|HURRICANE KATRINA EVACUEES', na=False)]

# Filter rows for bounds of Idaho
df_filtered = df_filtered[~((df_filtered['incidentBeginDate'].dt.year == 2005) & 
                            (df_filtered['Y'] >= min_lat) & (df_filtered['Y'] <= max_lat) & 
                            (df_filtered['X'] >= min_lon) & (df_filtered['X'] <= max_lon))]

# Save the filtered dataframe to a new CSV
df_filtered.to_csv(os.path.join(folder_path, 'UpdatedDisasterDeclarationSummariesPlusLocation_RowColFiltered.csv'), index=False)
