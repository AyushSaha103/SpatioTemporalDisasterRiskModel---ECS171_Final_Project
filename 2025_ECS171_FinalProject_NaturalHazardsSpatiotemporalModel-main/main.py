
import numpy as np
import pandas as pd
from DisasterDataset_3D_Visualizer import generateInteractive_3D_NodePlot

import sys
sys.path.append("helperFunctions/")
from ArtificiallyClusteredFilteredDatasetLoader import getClusteredDataset


# this script file is finished.
# it calls other functions from other scripts, which are also finished.


def logbase_n(val, logbase=3):
    return np.log(val) / np.log(logbase)

def main():
    print()

    # get dataset
    df = getClusteredDataset(inputDatasetPath="datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    sub_df_for_plotting = df[:len(df)//9]   # df[len(df)-int(4e3):]
    print(df.columns.tolist())

    #~~~~~~ plotting option 1: color-code the individual IDs for each plotted point ~~~~~~~
    
    # generate interactive 3D plot #1
    generateInteractive_3D_NodePlot(sub_df_for_plotting, colorShadedColumn="artificialClusterIndex")
    
    # generate interactive 3D plot #2
    generateInteractive_3D_NodePlot(sub_df_for_plotting, colorShadedColumn="incidentId_categoricalEncoding")
    


    #~~~~~~ plotting option 2: color-code the group sizes for each plotted point ~~~~~~~

    # # generate interactive 3D plot #1
    # generateInteractive_3D_NodePlot(sub_df_for_plotting, colorShadedColumn="artificialClusterGroupSize")
    
    # # generate interactive 3D plot #2
    # sub_df_for_plotting["commonIncidentIdGroupSize (logbase 1.5 scale)"] = logbase_n(sub_df_for_plotting["commonIncidentIdGroupSize"], 1.5)
    # generateInteractive_3D_NodePlot(sub_df_for_plotting, colorShadedColumn="commonIncidentIdGroupSize (logbase 1.5 scale)")
    
    print()

if __name__ == "__main__":
    main()

