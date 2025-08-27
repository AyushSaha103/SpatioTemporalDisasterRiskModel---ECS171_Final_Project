
# NOTE to all users/build-atop'ers: I recommend keeping the library imports below
# and following the folder structuring/formatting convention for all added components
# Good luck!



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



# TODO: put your prediction driver code here

def main():
    print("\nHello World...\n")

if __name__ == "__main__":
    main()
