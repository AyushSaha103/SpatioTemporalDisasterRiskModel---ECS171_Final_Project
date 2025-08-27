

# LIBRARY IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# library imports
import pandas as pd
import numpy as np
from pdb import set_trace as b
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from dateutil.relativedelta import relativedelta
import cartopy.crs as ccrs # for mapping states on plot
import cartopy.feature as cfeature

# local project imports

import sys
import os

# Add the parent directory to the path
helperFileFolders = ["DatasetLoaders/", "DatasetVisualization/", "GeneralHelpers/"]
for subFolder in helperFileFolders: sys.path.append(subFolder)

from ClusteredFilteredDatasetLoader import getArtificiallyClusteredDataset, getIncidentIdClusteredDataset
from DisasterDatasetVisualizers import *
from generalHelperFunctions import *
from FilteredDatasetLoader import retrieveRowColFilteredDataset



# this script file is finished.
# NOTE: it is a standalone script which uses (necessary) global vars, so other files should NOT import from this one.
# run this script to visualize a timelapsed animation of the disasters occuring.


# HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# return list of datetime objects ranging between tobj1 and tobj2, spaced by the user-defined amount
def generate_time_steps(tobj1, tobj2, years=0, months=0, days=0, hours=0):
    current_time = tobj1
    deltas = []
    while current_time <= tobj2:
        deltas.append(current_time)  # Store timedelta relative to tobj1
        current_time += relativedelta(years=years, months=months, days=days, hours=hours)
    return deltas

# return a custom-decremented datetime object
def decrement_timestamp(timestamp_obj, years=0, months=0, days=0):
    return timestamp_obj - relativedelta(years=years, months=months, days=days)
# RETIEVE DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = retrieveRowColFilteredDataset(inputDatasetPath="'../../datasets/Funding_DisasterDeclarationSummariesPlusLocation_RowColFiltered.csv'")
df["incidentId_categoricalEncoding (logbase 1.5 scale)"] = logbase_n(df["incidentId_categoricalEncoding"], 1.5)


# BASIC PRE-PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# basic dataset feature transformations
df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'], errors='coerce')
df['incidentEndDate'] = pd.to_datetime(df['incidentEndDate'], errors='coerce')

# create dummy values for sigma_x and sigma_y
df["sigma_x"] = np.random.uniform(0.1, 0.5, len(df))
df["sigma_y"] = np.random.uniform(0.1, 0.5, len(df))

# dummy values for sigma_t
df['sigma_t'] = np.random.randint(5, 30, len(df))  # Assign random durations (30-180 days)

# create time start and end position
df['sigma_t_start'] = df['incidentBeginDate'] - pd.to_timedelta(df['sigma_t'], unit='D')
df['sigma_t_end'] = df['incidentBeginDate'] + pd.to_timedelta(df['sigma_t'], unit='D')

# To make loading simpler, just sample a subset of the dataset
# Comment out for complete animation, but takes a long time to load.
df = df.sample(1000)

# DATASET ANIMATION - setting up color-coding parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

colorShadedColumnName = "incidentId_categoricalEncoding"    # options: "incidentIdClusterIndex", "incidentIdClusterGroupSize", "incidentId_categoricalEncoding", "artificialClusterIndex", "incidentId_categoricalEncoding (logbase 1.5 scale)", etc.
colorShadedColMin = min(df[colorShadedColumnName])
colorShadedColMax = max(df[colorShadedColumnName])

# DATASET ANIMATION - get the plotting timestamps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # get a set of time-steps (option 1 - just all the incidents' begin dates)
# unique_dates = df['incidentBeginDate'].unique()

# # get a set of time steps (option 2 - just all the incidents' begin and end dates)
# unique_dates = np.sort(np.union1d(df['incidentBeginDate'].dropna().unique(), \
#                                   df['incidentEndDate'].dropna().unique()))

# get a set of time-steps (option 3 - more realistically spaced time points, spaced by months=2 apart)
all_unique_dates = np.sort(np.union1d(df['incidentBeginDate'].dropna().unique(), df['incidentEndDate'].dropna().unique()))
unique_dates = generate_time_steps(all_unique_dates[0], all_unique_dates[-1], months=2)
SLIDING_TIME_WINDOW_DURATION_IN_YEARS = 9


# DATASET ANIMATION - set up the plot sizing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set up the plot size
x_min, x_max, y_min, y_max = min(df['X']), max(df['X']), min(df['Y']), max(df['Y'])
projection = ccrs.PlateCarree() # adding map features
fig, ax = plt.subplots(figsize=(18, 12), subplot_kw={'projection': projection})

# Add coastlines
# ax.coastlines(linewidth=0.5)

# Add U.S. state borders
# ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='black', linewidth=0.5)

# DATASET ANIMATION - plot update function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def basicPlotUpdater(frame):
    # PART 1 ---------------------------------------------------------------------------------------------------
    # Description: Various Options for extracting a sub_df to plot, corresponding to the current animation frame


    # # this option plots all prior disasters to each timestep, thereby clouding up the plot
    # sub_df = df[df['incidentBeginDate'] <= unique_dates[frame]]

    # # this option is very similar to ^
    # sub_df = df[(df['incidentEndDate'] <= unique_dates[frame])]

    # # this is the most proper option for representing the lifetime of each disaster, considering both begin & end date
    # sub_df = df[(df['incidentBeginDate'] <= unique_dates[frame]) & \
    #             (unique_dates[frame] <= df['incidentEndDate'])]
    
    # # this option plots all incidents from the prior __ years, for each animation frame
    sub_df = df[(df['incidentBeginDate'] <= unique_dates[frame]) & \
             (df['incidentBeginDate'] >= decrement_timestamp(unique_dates[frame], years=SLIDING_TIME_WINDOW_DURATION_IN_YEARS))]

    # # this option plots points according to the corresponding sigma_t value
    # curr_time = unique_dates[frame]
    # sub_df = df[(df['sigma_t_start'] <= curr_time) & (df['sigma_t_end'] >= curr_time)]
    
    # PART 2: create the scatter plot ---------------------------------------------------------------------------------------------------
    
    ax.clear()  # optional clearing of axis (prevents frame overlap)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # # this option just plots all points as standard blue color
    # scatter = ax.scatter(sub_df['X'], sub_df['Y'])

    # this option color-codes the plotted points by some dataframe column (may be slower)
    scatter = ax.scatter(sub_df['X'], sub_df['Y'], c=sub_df[colorShadedColumnName],
                         cmap='viridis', vmin=colorShadedColMin, vmax=colorShadedColMax)

    ax.set_title(f"Scatter Plot of disaster occurences for date: {unique_dates[frame]}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # plot the ellipse
    colormap = plt.get_cmap('viridis')
    for _, row in sub_df.iterrows():
        ih_funding = row['ihProgramDeclared']
        ia_funding = row['iaProgramDeclared']
        pa_funding = row['paProgramDeclared']
        hm_funding = row['hmProgramDeclared']
        
        funding_value = 0
        if ih_funding == 1:
            funding_value += 1
        if ia_funding == 1:
            funding_value += 2
        if pa_funding == 1:
            funding_value += 5
        if hm_funding == 1:
            funding_value += 8

        ellipse_area = funding_value/3  # Scale the funding value to get the ellipse area
        ellipse_width = np.sqrt(ellipse_area / np.pi) * 2  # Calculate width from area
        ellipse_height = np.sqrt(ellipse_area / np.pi) * 2  # Calculate height from area
        ellipse = Ellipse((row['X'], row['Y']), width=ellipse_width, height=ellipse_height,
                           linewidth=1.5, alpha=0.7)
        ax.add_patch(ellipse)

    # Add coastlines
    ax.coastlines(linewidth=0.5)

    # Add U.S. state borders
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='black', linewidth=0.5)

    return scatter,

if __name__ == '__main__':
    df_small = df.sample(1000)
    print("Running the disaster dataset timelapse animator...")
    ani = FuncAnimation(fig, basicPlotUpdater, frames=len(unique_dates), interval=120, repeat=False)
    ani.save('Funding-Severity_Dataset_Visualizer.mp4', writer='ffmpeg', fps=30)

    plt.show()
    print("DONE")
