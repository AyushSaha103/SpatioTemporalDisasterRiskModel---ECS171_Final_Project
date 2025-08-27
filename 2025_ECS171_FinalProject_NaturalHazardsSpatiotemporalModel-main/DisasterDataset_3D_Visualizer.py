
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import sys
sys.path.append("helperFunctions/")
from ArtificiallyClusteredFilteredDatasetLoader import getClusteredDataset

# this script file is finished
# run it to view an interactive 3D plot of all the disaster occurences in a space/time grid
# NOTE: computationally expensive

def logbase_n(val, logbase=3):
    return np.log(val) / np.log(logbase)

def plotIncidentIdGroupSizesDistribution(df):
    # Count number of rows per unique incidentID
    subgraph_sizes = df["incidentId"].value_counts()

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(subgraph_sizes, bins=range(1, subgraph_sizes.max() + 1), alpha=0.7, edgecolor="black")
    plt.xlabel("Common-IncidentId-Group Size")
    plt.ylabel("Number of NODE GROUPS")
    plt.title("Distribution of Common-IncidentId-Group Sizes")
    # plt.yscale("log")
    plt.show()
    # plt.savefig("testingPlots/subgraphSizesDistr.png")

def plotWeightedIncidentIdGroupSizesDistribution(df):
    # Count occurrences of each incidentId (subgraph size)
    subgraph_sizes = df["incidentId"].value_counts()

    # Convert to dictionary {subgraph size: count}
    subgraph_size_counts = subgraph_sizes.value_counts().to_dict()

    # Multiply count by subgraph size for weighted frequency
    weighted_counts = {size: size * count for size, count in subgraph_size_counts.items()}

    # Plot weighted histogram
    plt.figure(figsize=(8, 5))
    plt.bar(weighted_counts.keys(), weighted_counts.values(), edgecolor="black", alpha=0.7)
    plt.xlabel("Common-IncidentId-Group Size")
    plt.ylabel("Number of NODES")
    plt.title("Weighted Distribution of Common-IncidentId-Group Sizes")

    plt.show()


def generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize"):
    fig = px.scatter_3d(df, x="X", y="Y", z="timeInDays", color=colorShadedColumn,
                        title="3D Scatter Plot of Hazard Incidents",
                        labels={"X": "Longitude", "Y": "Latitude", "timeInDays": "Days Since First Incident"},
                        opacity=0.8)
    fig.show()


def main():
    # get dataset
    print()
    df = getClusteredDataset(inputDatasetPath="datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    print("Retrieved row/column-filtered dataset!")

    # interactive 3D plot (opens automatically in browser)
    # NOTE: computationally expensive
    print("Generating visualization plots...")
    df["commonIncidentIdGroupSize (logbase 1.5 scale)"] = logbase_n(df["commonIncidentIdGroupSize"], 1.5)
    generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize (logbase 1.5 scale)")
    # generateInteractive_3D_NodePlot(df, colorShadedColumn="artificialClusterGroupSize")
    
    # histogram of count of different-sized common-incidentId-subgroups
    plotIncidentIdGroupSizesDistribution(df)

    # histogram of weighted counts of different-sized common-incidentId-subgroups
    # the count of each subgraph size is multiplied by its (the subgraph's) size
    plotWeightedIncidentIdGroupSizesDistribution(df) 

    print()

if __name__ == "__main__":
    main()




