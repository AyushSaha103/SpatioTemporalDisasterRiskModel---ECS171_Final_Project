

# LIBRARY IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dateutil.relativedelta import relativedelta
from scipy.stats import gaussian_kde
import geopandas as gpd
us_shapefile_path = "datasets/ne_110m_admin_0_countries.shp"
world = gpd.read_file(us_shapefile_path)
us = world[world['SOVEREIGNT'] == 'United States of America']

import sys
# sys.path.append("helperFunctions/")
# from DatasetLoader import *
helperFileFolders = ["DatasetLoaders/", "DatasetVisualization/", "GeneralHelpers/"]
for subFolder in helperFileFolders: sys.path.append(subFolder)

from ClusteredFilteredDatasetLoader import getArtificiallyClusteredDataset, getIncidentIdClusteredDataset
from DisasterDatasetVisualizers import *
from generalHelperFunctions import *



# HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# return list of datetime objects ranging between tobj1 and tobj2, spaced by the user-defined amount
def generate_time_steps(tobj1, tobj2, years=0, months=0, days=0, hours=0):
    current_time = tobj1
    deltas = []
    while current_time <= tobj2:
        deltas.append(current_time)  # Store timedelta relative to tobj1
        current_time += relativedelta(years=years, months=months, days=days, hours=hours)
    return deltas

# return a custom-incremented datetime object
def increment_timestamp(timestamp_obj, years=0, months=0, days=0):
    return timestamp_obj + relativedelta(years=years, months=months, days=days)

# return a custom-decremented datetime object
def decrement_timestamp(timestamp_obj, years=0, months=0, days=0):
    return timestamp_obj - relativedelta(years=years, months=months, days=days)



# RETIEVE DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
df["incidentId_categoricalEncoding (logbase 1.5 scale)"] = logbase_n(df["incidentId_categoricalEncoding"], 1.5)


# BASIC PRE-PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# basic dataset feature transformations
df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'], errors='coerce')
df['incidentEndDate'] = pd.to_datetime(df['incidentEndDate'], errors='coerce')

print("columns: ", df.columns)


# MORE ADVANCED PRE-PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# we probably need to:
#   - either normalize all the XY coordinates or squish them between 0 & 1
#   - (maybe) convert the time variable to a floating-point format and normalize this



# DATASET ANIMATION - get the plotting timestamps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # get a set of time-steps (option 1 - just all the incidents' begin dates)
# unique_dates = df['incidentBeginDate'].unique()

# # get a set of time steps (option 2 - just all the incidents' begin and end dates)
# unique_dates = np.sort(np.union1d(df['incidentBeginDate'].dropna().unique(), \
#                                   df['incidentEndDate'].dropna().unique()))

# get a set of time-steps (option 3 - more realistically spaced time points, but only considers events w/ duration > [months=2])
all_unique_dates = np.sort(np.union1d(df['incidentBeginDate'].dropna().unique(), df['incidentEndDate'].dropna().unique()))
unique_dates = generate_time_steps(all_unique_dates[0], all_unique_dates[-1], months=2)



# DATASET ANIMATION - set up the plot stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

colorShadedColumnName = "incidentId_categoricalEncoding"    # options: "incidentIdClusterIndex", "incidentIdClusterGroupSize", "incidentId_categoricalEncoding", "artificialClusterIndex", "incidentId_categoricalEncoding (logbase 1.5 scale)", etc.
colorShadedColMin = min(df[colorShadedColumnName])
colorShadedColMax = max(df[colorShadedColumnName])

fig, ax = plt.subplots(figsize=(18, 12))
# Filter the data to be within the bounds of the continental US
x_min, x_max = -125, -66
y_min, y_max = 24.396308, 49.384358
us_df = df[(df['Y'] >= y_min) & (df['Y'] <= y_max) & (df['X'] >= x_min) & (df['X'] <= x_max)]

# get a set of time-steps (option 3 - more realistically spaced time points, spaced by months=2 apart)
all_unique_dates = np.sort(np.union1d(df['incidentBeginDate'].dropna().unique(), df['incidentEndDate'].dropna().unique()))
unique_dates = generate_time_steps(all_unique_dates[0], all_unique_dates[-1], months=2)
SLIDING_TIME_WINDOW_DURATION_IN_YEARS = 9



# DATASET ANIMATION - plot update function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def basicPlotUpdater(frame): 
    sub_df = us_df[(df['incidentBeginDate'] <= unique_dates[frame]) & \
            (us_df['incidentBeginDate'] >= decrement_timestamp(unique_dates[frame], years=SLIDING_TIME_WINDOW_DURATION_IN_YEARS))]
    
    # get rid of evacuation entries, only want entries where disaster actually happened
    sub_df = sub_df[~sub_df['declarationTitle'].str.endswith('EVACUATION', na=False)]
    sub_df = sub_df[~sub_df['declarationTitle'].str.endswith('EVACUEES', na=False)]

    # Exclude data points from Idaho in 2005
    sub_df = sub_df[~((sub_df['incidentBeginDate'].dt.year == 2005) &
                    (sub_df['Y'] >= 41.9880) & (sub_df['Y'] <= 49.0030) &  # Latitude range of Idaho
                    (sub_df['X'] >= -117.2437) & (sub_df['X'] <= -111.0430))]  # Longitude range of Idaho
    
    disaster_type_colors = {
        "storm": "red",
        "flood": "blue",
        "rain": "cyan",
        "ice": "purple",
        "snow": "lightblue",
        "blizzard": "darkblue",
        "hurricane": "orange",
    }

    sub_df['color'] = sub_df['declarationTitle'].apply(lambda title: next((color for disaster, color in disaster_type_colors.items() if disaster in title.lower()), 'gray'))

    # PART 2: create the scatter plot ---------------------------------------------------------------------------------------------------
    
    ax.clear()

    us.plot(ax=ax, color='lightgray', edgecolor='black', zorder=1)
    ax.set_xlim(x_min, x_max) 
    ax.set_ylim(y_min, y_max)

    scatter = ax.scatter(sub_df['X'], sub_df['Y'], c=sub_df['color'], zorder=2)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=disaster, 
                          markersize=10, markerfacecolor=color) 
               for disaster, color in disaster_type_colors.items()]

    ax.legend(handles=handles, title='Disaster Type', loc='upper right', fontsize=12)

    ax.set_title(f"Scatter Plot of disaster occurences for date: {unique_dates[frame]}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return scatter,

ani = FuncAnimation(fig, basicPlotUpdater, frames=len(unique_dates), interval=120, repeat=False)

plt.show()
print("DONE")
