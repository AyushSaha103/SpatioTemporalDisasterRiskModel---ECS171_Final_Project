
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pdb import set_trace as b

# this script file is finished

def logbase_n(val, logbase=3):
    return np.log(val) / np.log(logbase)

def plotAnySubgroupSizesDistribution(df, colName="incidentId"):
    # Count number of rows per unique incidentID
    subgroup_sizes = df[colName].value_counts()

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(subgroup_sizes, bins=range(1, subgroup_sizes.max() + 1), alpha=0.7, edgecolor="black")
    plt.xlabel(f"Common {colName} Group Size")
    plt.ylabel("Number of NODE GROUPS")
    plt.title(f"Distribution of Common {colName} Group Sizes")
    # plt.yscale("log")
    plt.show()
    # plt.savefig("testingPlots/subgraphSizesDistr.png")

def plotAnySubgroupSizesDistribution_Weighted(df, colName="incidentId"):
    # Count occurrences of each incidentId (subgraph size)
    subgraph_sizes = df[colName].value_counts()

    # Convert to dictionary {subgraph size: count}
    subgraph_size_counts = subgraph_sizes.value_counts().to_dict()

    # Multiply count by subgraph size for weighted frequency
    weighted_counts = {size: size * count for size, count in subgraph_size_counts.items()}

    # Plot weighted histogram
    plt.figure(figsize=(8, 5))
    plt.bar(weighted_counts.keys(), weighted_counts.values(), edgecolor="black", alpha=0.7)

    plt.xlabel(f"Common {colName} Group Size")
    plt.ylabel("Number of NODES")
    plt.title(f"Weighted Distribution of Common {colName} Group Sizes")

    plt.show()

def generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize"):
    fig = px.scatter_3d(df, x="X", y="Y", z="timeOfStartInDays", color=colorShadedColumn,
                        title="3D Scatter Plot of Hazard Incidents",
                        labels={"X": "Longitude", "Y": "Latitude", "timeOfStartInDays": "Days Since First Incident"},
                        opacity=0.8)
    fig.show()




import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pdb import set_trace as b

# this script file is finished

def logbase_n(val, logbase=3):
    return np.log(val) / np.log(logbase)

def plotAnySubgroupSizesDistribution(df, colName="incidentId"):
    # Count number of rows per unique incidentID
    subgroup_sizes = df[colName].value_counts()

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(subgroup_sizes, bins=range(1, subgroup_sizes.max() + 1), alpha=0.7, edgecolor="black")
    plt.xlabel(f"Common {colName} Group Size")
    plt.ylabel("Number of NODE GROUPS")
    plt.title(f"Distribution of Common {colName} Group Sizes")
    # plt.yscale("log")
    plt.show()
    # plt.savefig("testingPlots/subgraphSizesDistr.png")

def plotAnySubgroupSizesDistribution_Weighted(df, colName="incidentId"):
    # Count occurrences of each incidentId (subgraph size)
    subgraph_sizes = df[colName].value_counts()

    # Convert to dictionary {subgraph size: count}
    subgraph_size_counts = subgraph_sizes.value_counts().to_dict()

    # Multiply count by subgraph size for weighted frequency
    weighted_counts = {size: size * count for size, count in subgraph_size_counts.items()}

    # Plot weighted histogram
    plt.figure(figsize=(8, 5))
    plt.bar(weighted_counts.keys(), weighted_counts.values(), edgecolor="black", alpha=0.7)

    plt.xlabel(f"Common {colName} Group Size")
    plt.ylabel("Number of NODES")
    plt.title(f"Weighted Distribution of Common {colName} Group Sizes")

    plt.show()

def generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize"):
    fig = px.scatter_3d(df, x="X", y="Y", z="timeOfStartInDays", color=colorShadedColumn,
                        title="3D Scatter Plot of Hazard Incidents",
                        labels={"X": "Longitude", "Y": "Latitude", "timeOfStartInDays": "Days Since First Incident"},
                        opacity=0.8)
    fig.show()


