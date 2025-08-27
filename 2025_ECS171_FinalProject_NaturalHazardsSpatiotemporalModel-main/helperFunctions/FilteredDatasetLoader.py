
import pandas as pd
from functools import reduce
import numpy as np
import os
from pdb import set_trace as b

# this script file is finished
# OUTPUT: filtered dataframe, with these new columns:
#   -> "timeInDays" since the very first disaster in the dataset
#   -> one-hot encoded columns titled "disasterType_<some type>", informing which disaster types they were
#   -> "disasterType_categoricalEncoding" column, representing the unique integer assigned to represent each row's one-hot encoded vector of disasterType categories
#       this is needed to differentiate the one-hot encoded rows which belong to multiple classes
# TO GET OUTPUT, call function: retrieveRowColFilteredDataset(datafilePath, disasterTypesOfInterest, enableLoadingSavedFilteredFile)



# helper functions

def getDisasterTypeFilteredDataFrame(df, disasterType):
    return df[(df["declarationTitle"].str.lower().str.contains(disasterType, na=False))]

def displayNumOccurencesOfAllDisasterTypes(df):
    for val in  ["N/A", "fire", "tornado", "tsunami", "snow", "ice", "rain", "landslide", "flood", "blizzard", "wind", "thunderstorm", "storm", "earthquake", "hurricane", "temperature", "freez"]:
        count = len(df[(df["declarationTitle"].str.lower().str.contains(val, na=False))])
        print(f"# occurences of {val} in dataset: {count}")


def getContinuousTimeColumn(df):
    incidentBeginDates = pd.to_datetime(df["incidentBeginDate"])
    # Normalize the time axis (set the earliest date to 0)
    min_date = incidentBeginDates.min()
    timeVals = (incidentBeginDates - min_date).dt.total_seconds() / (24 * 3600)  # Convert to days
    return timeVals

def get_subgraph_size_column(df):
    return df["incidentId"].map(df["incidentId"].value_counts())

def getCategoricallyEncodedIncidentIdColumn(df):
    unique_ids = df['incidentId'].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids))}
    return df['incidentId'].map(id_mapping)

def getDisasterTypeNumericIdentifierColumn(df):
    disaster_cols = [col for col in df.columns if col.startswith("disasterType_")]
    disasterTypesDataframe = df[disaster_cols].apply(lambda row: tuple(row.values), axis=1)
    unique_combinations = {combo: i for i, combo in enumerate(disasterTypesDataframe.unique())}
    return disasterTypesDataframe.map(unique_combinations).astype(float)


# MOST IMPORTANT FUNCTION in file: OUR DATAFRAME-RETRIEVER

def retrieveRowColFilteredDataset(inputDatasetPath = "datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                                  disasterTypesOfInterest = ["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],
                                  enableLoadingSavedFilteredFile=False,
                                  outputFilteredDatafilePath = "datasets/DisasterDeclarationSummariesPlusLocation_RowColFiltered.csv"):
    # NOTE: all possible disasterTypesOfInterest are ["fire", "tornado", "tsunami", "snow", "ice", "rain", "landslide", "flood", "blizzard", "wind", "thunderstorm", "storm", "earthquake", "hurricane", "temperature", "freeze"]

    # first try to load the pre-filtered dataset (if exists)
    if enableLoadingSavedFilteredFile and os.path.exists(outputFilteredDatafilePath):
        df = pd.read_csv(outputFilteredDatafilePath, index_col=None)
        displayNumOccurencesOfAllDisasterTypes(df)
        print(f"de-nullified dataset length (filtered by disaster type={disasterTypesOfInterest}): {len(df)}\n")
        return df

    # read in dataset
    df = pd.read_csv(inputDatasetPath, index_col=None)
    print("original dataset length: " + str(len(df)))
    displayNumOccurencesOfAllDisasterTypes(df)

    # # filter df to a certain disaster type
    # disasterTypeOfInterest = "storm"        # ["fire", "tornado", "tsunami", "snow", "ice", "rain", "landslide", "flood", "blizzard", "wind", "thunderstorm", "storm", "earthquake", "hurricane", "temperature", "freeze"]
    # df = df[(df["declarationTitle"].str.lower().str.contains(disasterTypeOfInterest, na=False))]
    # print(f"\nrow-filtered dataset length (filtered by disaster type={disasterTypeOfInterest}): {len(df)}")

    # filter df to a subset of disaster types
    dfs = [getDisasterTypeFilteredDataFrame(df, disasterType) for disasterType in disasterTypesOfInterest]
    df = pd.concat(dfs, axis=0)
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df['X'] > -130) & (df['X'] < -45) & (df['Y'] > 20) & (df['Y'] < 55)]
    print(f"row-filtered dataset length (filtered by disaster types in {disasterTypesOfInterest}): {len(df)}")

    # remove null rows
    df = df[["incidentBeginDate", "incidentEndDate", "X", "Y", "declarationTitle", "incidentId"]]
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"de-nullified dataset length: {len(df)}\n")

    # column filtering and adding
    df["incidentId_categoricalEncoding"] = getCategoricallyEncodedIncidentIdColumn(df)
    df["commonIncidentIdGroupSize"] = get_subgraph_size_column(df)
    df["timeInDays"] = getContinuousTimeColumn(df)
    df = df.sort_values(by="timeInDays")

    # add one-hot encoded disasterType cols (and a numeric identifier column)
    for disasterType in disasterTypesOfInterest:
        df[f"disasterType_{disasterType}"] = df["declarationTitle"].str.lower().str.contains(disasterType).astype(int)
    df["disasterType_categoricalEncoding"] = getDisasterTypeNumericIdentifierColumn(df)

    # save the filtered dataset to csv
    # if not os.path.exists(filteredDatafilePath):
    df.to_csv(outputFilteredDatafilePath, index=False)
    print(f"Saved row/col-filtered dataframe to csv file {outputFilteredDatafilePath}")
    return df

# driver code

def main():
    print()
    
    df = retrieveRowColFilteredDataset()
    print("Retrieved row/column-filtered dataframe!")
    print(f"dataframe columns: {df.columns.tolist()}")
    print()
    print("dataframe head:")
    print(df.head())

    print()

if __name__ == "__main__":
    main()