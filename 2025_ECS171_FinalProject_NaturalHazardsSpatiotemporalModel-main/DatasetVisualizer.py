
# library imports

import numpy as np
import pandas as pd
from pdb import set_trace as b

# local project imports

import sys
helperFileFolders = ["DatasetLoaders/", "DatasetVisualization/", "GeneralHelpers/"]
for subFolder in helperFileFolders: sys.path.append(subFolder)

from ClusteredFilteredDatasetLoader import getArtificiallyClusteredDataset, getIncidentIdClusteredDataset
from DisasterDatasetVisualizers import *
from generalHelperFunctions import *


# this script file is finished.
# it calls other functions from other scripts, which are also finished.


def main():
    # # RETRIEVE DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print()
    # df = getArtificiallyClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    df = getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    print("Retrieved clustered dataset!")

    # interactive 3D plot (opens automatically in browser)
    # NOTE: computationally expensive
    print("Generating visualization plots...")
    
    # # INTERACTIVE 3D PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # option 1
    # df["commonIncidentIdGroupSize (logbase 1.5 scale)"] = logbase_n(df["commonIncidentIdGroupSize"], 1.5)
    # generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize (logbase 1.5 scale)")

    # # option 2
    # generateInteractive_3D_NodePlot(df, colorShadedColumn="incidentId_categoricalEncoding")

    # option 3
    generateInteractive_3D_NodePlot(df, colorShadedColumn="incidentIdClusterIndex")
    
    # # option 4
    # generateInteractive_3D_NodePlot(df, colorShadedColumn="artificialClusterIndex")


    # # HISTOGRAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # histogram of counts of different-sized common-incidentIdCluster-subgroups
    plotAnySubgroupSizesDistribution(df, "incidentIdClusterIndex")

    # histogram of count of different-sized common-incidentId-subgroups
    plotAnySubgroupSizesDistribution(df, "incidentId")

    # histogram of weighted counts of different-sized common-incidentId-subgroups
    # the count of each subgraph size is multiplied by its (the subgraph's) size
    plotAnySubgroupSizesDistribution_Weighted(df, "incidentId") 

    print()

if __name__ == "__main__":
    main()
