


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pdb import set_trace as b
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warning

from FilteredDatasetLoader import retrieveRowColFilteredDataset

# this script file is finished and provides these two dataset retriever functions:

# getArtificiallyClusteredDataset()
# OUTPUT: filtered AND clustered dataframe, with these new columns:
#   -> "artificialClusterIndex", an integer index denoting the assigned spatiotemporal cluster group which each data row (node) belongs to
#   -> "artificialClusterGroupSize", an integer value denoting the size of each data row's respective cluster group
#   NOTE: the clustered groups have a max size of maxClusterSize nodes

# getIncidentIdClusteredDataset()
# OUTPUT: filtered AND clustered dataframe, with these new columns:
#   -> "incidentIdClusterIndex", an integer index denoting the assigned spatiotemporal cluster group which each data row (node) belongs to
#   -> "incidentIdClusterGroupSize", an integer value denoting the size of each data row's respective cluster group
#   NOTE: the clustered groups have a max size of maxClusterSize nodes


class DataNormalizer:
    def __init__(self):
        self.scaler_xy = StandardScaler()
        self.scaler_xyt = MinMaxScaler()
        self.timeUpscalingRatio = 1.0

    def normalize(self, df, timeFeatureImportanceRatio=1.0):
        df[["X", "Y"]] = self.scaler_xy.fit_transform(df[["X", "Y"]])
        df[["X", "Y", "timeOfStartInDays"]] = self.scaler_xyt.fit_transform(df[["X","Y","timeOfStartInDays"]])
        
        self.timeUpscalingRatio = 1 / timeFeatureImportanceRatio
        df["timeOfStartInDays"] *= self.timeUpscalingRatio
        return df
    
    def denormalize(self, df):
        df["timeOfStartInDays"] /= self.timeUpscalingRatio
        df[["X", "Y", "timeOfStartInDays"]] = self.scaler_xyt.inverse_transform(df[["X","Y","timeOfStartInDays"]])
        df[["X", "Y"]] = self.scaler_xy.inverse_transform(df[["X", "Y"]])
        return df

def getRandomlySortedValsOfCategoricallyEncodedColumn(df, columnName):
    unique_clusters = df[columnName].unique()
    randomized_clusters = np.random.permutation(unique_clusters)
    cluster_mapping = dict(zip(unique_clusters, randomized_clusters))
    newColVals = df[columnName].map(cluster_mapping)
    return newColVals, cluster_mapping

def showHistogramOfSubgraphSizes(subgraph_sizes):
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(subgraph_sizes, bins=range(min(subgraph_sizes), max(subgraph_sizes) + 2), edgecolor='black', align='left')

    # Labels and title
    plt.xlabel("Subgraph Size")
    plt.ylabel("Count")
    plt.title("Distribution of Clustered Subgraph Sizes")
    plt.xticks(range(min(subgraph_sizes), max(subgraph_sizes) + 1))

    # Show the plot
    plt.show()

def getKMeansClusterIndexColumn(sub_df, n_clusters=2, availableClusterIndex=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    assignedClusterCol = kmeans.fit_predict(sub_df[['X', 'Y', "timeOfStartInDays"]])
    assignedClusterCol += availableClusterIndex
    return assignedClusterCol

# returns the passed in dataframe "df", with a new column added named by clusterIndexColName
def getClusteredDataBatch(df, max_cluster_size, method='ward', allowOversizedClusters=False, clusterIndexColName="artificialClusterIndex"):
    if len(df) <= max_cluster_size:
        df[clusterIndexColName] = [0]*len(df)
        return df

    # run first stage of hierarchical clustering procedure
    data = df[['X', 'Y', "timeOfStartInDays"]].values
    Z = linkage(data, method=method)
    num_clusters = len(df) // max_cluster_size
    df[clusterIndexColName] = fcluster(Z, num_clusters, criterion='maxclust')
    cluster_sizes = df[clusterIndexColName].value_counts()

    if allowOversizedClusters:
        return df
    
    # split up oversized clusters into smaller ones
    subsplit_count = 0
    while cluster_sizes.max() > max_cluster_size:
        cluster_sizes = df[clusterIndexColName].value_counts()
        if cluster_sizes.max() <= max_cluster_size:
            break

        largest_cluster = cluster_sizes.idxmax()
        sub_df = df[df[clusterIndexColName] == largest_cluster]
        num_sub_clusters = int(len(sub_df) // max_cluster_size) + 1
        availableClusterIndex = df[clusterIndexColName].max() + 1

        print(f"Splitting cluster of size {len(sub_df)} into {num_sub_clusters} sub-clusters")
        new_clusters = getKMeansClusterIndexColumn(sub_df, num_sub_clusters, availableClusterIndex)
        df.loc[sub_df.index, clusterIndexColName] = new_clusters

        subsplit_count += 1

    # "normalize" the cluster values column
    df[clusterIndexColName] = pd.factorize(df[clusterIndexColName])[0]

    # # display histogram of the subgraph sizes
    # showHistogramOfSubgraphSizes(df[clusterIndexColName].value_counts())
    
    return df

def displaySizingInfoAboutDataset(df):
    print(f"Dataset size (# of rows): {len(df)}")
    print(f"dataframe columns: {df.columns.tolist()}")
    print()
    print("dataframe head:")
    print(df.head())
    print()

# will return a dataframe with 2 new columns added: artificialClusterIndex, artificialClusterGroupSize
def getArtificiallyClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                        disasterTypesOfInterest=["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],
                        dataClusteringBatchSize=6221, maxClusterSize=40, timeFeatureImportanceRatio=(1/6),
                        enableLoadingSavedOutputFile=True,
                        outputClusteredDatasetPath="Datasets/DisasterDeclarationSummariesPlusLocation_ArtificiallyClustered.csv"):
    
    # first try to read in the pre-saved clustered dataset (if exists)
    if enableLoadingSavedOutputFile and os.path.exists(outputClusteredDatasetPath):
        df = pd.read_csv(outputClusteredDatasetPath, index_col=None)
        displaySizingInfoAboutDataset(df)
        return df

    # retrieve the row/column-diltered dataset
    df = retrieveRowColFilteredDataset(inputDatasetPath = inputDatasetPath,
                                       disasterTypesOfInterest=disasterTypesOfInterest)
    
    clusteredBatches = []
    clusterIndexIncrement = 0
    for i in range(0, len(df), dataClusteringBatchSize):
        sub_df = df[i:min(len(df), i+dataClusteringBatchSize)]

        # normalize sub_df
        normalizer = DataNormalizer()
        sub_df = normalizer.normalize(sub_df, timeFeatureImportanceRatio=timeFeatureImportanceRatio)
        
        # cluster dataset, add columns "artificialClusterIndex" and "artificialClusterGroupSize" to dataset
        sub_df = getClusteredDataBatch(sub_df, max_cluster_size=maxClusterSize, clusterIndexColName="artificialClusterIndex")
        sub_df['artificialClusterGroupSize'] = sub_df["artificialClusterIndex"].map(sub_df["artificialClusterIndex"].value_counts())

        sub_df["artificialClusterIndex"] += clusterIndexIncrement
        clusterIndexIncrement = max(sub_df["artificialClusterIndex"]) + 1

        # denormalize sub_df and append it to clusteredBatches
        sub_df = normalizer.denormalize(sub_df)
        clusteredBatches.append(sub_df)
    print("Finished splitting all clusters!\n")
    
    # retain original dataframe, now with a nice "artificialClusterIndex" column added denoting the cluster each row belongs to
    df = pd.concat(clusteredBatches, axis=0)
    df["artificialClusterIndex"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "artificialClusterIndex")
    df["incidentId_categoricalEncoding"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentId_categoricalEncoding")
    
    df.to_csv(outputClusteredDatasetPath, index=False)
    displaySizingInfoAboutDataset(df)
    return df

# will return a dataframe with 2 new columns added: incidentIdClusterIndex, incidentIdClusterGroupSize
def getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                        disasterTypesOfInterest=["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],
                        dataClusteringMaxBatchSize=6221, maxClusterSize=40, timeFeatureImportanceRatio=(1/6),
                        enableLoadingSavedOutputFile=True,
                        outputClusteredDatasetPath="Datasets/DisasterDeclarationSummariesPlusLocation_IncidentIdClustered.csv"):
    
    # first try to read in the pre-saved clustered dataset (if exists)
    if enableLoadingSavedOutputFile and os.path.exists(outputClusteredDatasetPath):
        df = pd.read_csv(outputClusteredDatasetPath, index_col=None)
        displaySizingInfoAboutDataset(df)
        return df

    # retrieve the row/column-diltered dataset
    df = retrieveRowColFilteredDataset(inputDatasetPath = inputDatasetPath,
                                       disasterTypesOfInterest=disasterTypesOfInterest)
    
    clusteredBatches = []
    clusterIndexIncrement = 0
    uniqueIncidentIds = df["incidentId"].unique().tolist()

    for i in range(len(uniqueIncidentIds)):
        df_commonIncId = df[df["incidentId"]==uniqueIncidentIds[i]]
        for j in range(0, len(df_commonIncId), dataClusteringMaxBatchSize):
            sub_df = df_commonIncId[j:min(len(df_commonIncId), j+dataClusteringMaxBatchSize)]
            
            # normalize sub_df
            normalizer = DataNormalizer()
            sub_df = normalizer.normalize(sub_df, timeFeatureImportanceRatio=timeFeatureImportanceRatio)
        
            # cluster dataset, add columns "incidentIdClusterIndex" and "incidentIdClusterGroupSize" to dataset
            sub_df = getClusteredDataBatch(sub_df, max_cluster_size=maxClusterSize, clusterIndexColName="incidentIdClusterIndex")
            sub_df['incidentIdClusterGroupSize'] = sub_df["incidentIdClusterIndex"].map(sub_df["incidentIdClusterIndex"].value_counts())

            sub_df["incidentIdClusterIndex"] += clusterIndexIncrement
            clusterIndexIncrement = max(sub_df["incidentIdClusterIndex"]) + 1

            # denormalize sub_df and append it to clusteredBatches
            sub_df = normalizer.denormalize(sub_df)
            clusteredBatches.append(sub_df)
    print("Finished splitting all clusters!\n")
    
    # retain original dataframe, now with a nice "incidentIdClusterIndex" column added denoting the cluster each row belongs to
    df = pd.concat(clusteredBatches, axis=0)
    df["incidentIdClusterIndex"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentIdClusterIndex")
    df["incidentId_categoricalEncoding"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentId_categoricalEncoding")
    
    df.to_csv(outputClusteredDatasetPath, index=False)
    displaySizingInfoAboutDataset(df)
    return df




# driver code

def main():
    print()

    df = getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    # df = getArtificiallyClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    print("Finished retrieving clustered dataset!\n")

    # maxClusterSize = max(df["artificialClusterGroupSize"])
    # print(f"Max cluster group size: {maxClusterSize}")
    
    print()

if __name__ == "__main__":
    main()





from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pdb import set_trace as b
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warning

from FilteredDatasetLoader import retrieveRowColFilteredDataset

# this script file is finished and provides these two dataset retriever functions:

# getArtificiallyClusteredDataset()
# OUTPUT: filtered AND clustered dataframe, with these new columns:
#   -> "artificialClusterIndex", an integer index denoting the assigned spatiotemporal cluster group which each data row (node) belongs to
#   -> "artificialClusterGroupSize", an integer value denoting the size of each data row's respective cluster group
#   NOTE: the clustered groups have a max size of maxClusterSize nodes

# getIncidentIdClusteredDataset()
# OUTPUT: filtered AND clustered dataframe, with these new columns:
#   -> "incidentIdClusterIndex", an integer index denoting the assigned spatiotemporal cluster group which each data row (node) belongs to
#   -> "incidentIdClusterGroupSize", an integer value denoting the size of each data row's respective cluster group
#   NOTE: the clustered groups have a max size of maxClusterSize nodes


class DataNormalizer:
    def __init__(self):
        self.scaler_xy = StandardScaler()
        self.scaler_xyt = MinMaxScaler()
        self.timeUpscalingRatio = 1.0

    def normalize(self, df, timeFeatureImportanceRatio=1.0):
        df[["X", "Y"]] = self.scaler_xy.fit_transform(df[["X", "Y"]])
        df[["X", "Y", "timeOfStartInDays"]] = self.scaler_xyt.fit_transform(df[["X","Y","timeOfStartInDays"]])
        
        self.timeUpscalingRatio = 1 / timeFeatureImportanceRatio
        df["timeOfStartInDays"] *= self.timeUpscalingRatio
        return df
    
    def denormalize(self, df):
        df["timeOfStartInDays"] /= self.timeUpscalingRatio
        df[["X", "Y", "timeOfStartInDays"]] = self.scaler_xyt.inverse_transform(df[["X","Y","timeOfStartInDays"]])
        df[["X", "Y"]] = self.scaler_xy.inverse_transform(df[["X", "Y"]])
        return df

def getRandomlySortedValsOfCategoricallyEncodedColumn(df, columnName):
    unique_clusters = df[columnName].unique()
    randomized_clusters = np.random.permutation(unique_clusters)
    cluster_mapping = dict(zip(unique_clusters, randomized_clusters))
    newColVals = df[columnName].map(cluster_mapping)
    return newColVals, cluster_mapping

def showHistogramOfSubgraphSizes(subgraph_sizes):
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(subgraph_sizes, bins=range(min(subgraph_sizes), max(subgraph_sizes) + 2), edgecolor='black', align='left')

    # Labels and title
    plt.xlabel("Subgraph Size")
    plt.ylabel("Count")
    plt.title("Distribution of Clustered Subgraph Sizes")
    plt.xticks(range(min(subgraph_sizes), max(subgraph_sizes) + 1))

    # Show the plot
    plt.show()

def getKMeansClusterIndexColumn(sub_df, n_clusters=2, availableClusterIndex=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    assignedClusterCol = kmeans.fit_predict(sub_df[['X', 'Y', "timeOfStartInDays"]])
    assignedClusterCol += availableClusterIndex
    return assignedClusterCol

# returns the passed in dataframe "df", with a new column added named by clusterIndexColName
def getClusteredDataBatch(df, max_cluster_size, method='ward', allowOversizedClusters=False, clusterIndexColName="artificialClusterIndex"):
    if len(df) <= max_cluster_size:
        df[clusterIndexColName] = [0]*len(df)
        return df

    # run first stage of hierarchical clustering procedure
    data = df[['X', 'Y', "timeOfStartInDays"]].values
    Z = linkage(data, method=method)
    num_clusters = len(df) // max_cluster_size
    df[clusterIndexColName] = fcluster(Z, num_clusters, criterion='maxclust')
    cluster_sizes = df[clusterIndexColName].value_counts()

    if allowOversizedClusters:
        return df
    
    # split up oversized clusters into smaller ones
    subsplit_count = 0
    while cluster_sizes.max() > max_cluster_size:
        cluster_sizes = df[clusterIndexColName].value_counts()
        if cluster_sizes.max() <= max_cluster_size:
            break

        largest_cluster = cluster_sizes.idxmax()
        sub_df = df[df[clusterIndexColName] == largest_cluster]
        num_sub_clusters = int(len(sub_df) // max_cluster_size) + 1
        availableClusterIndex = df[clusterIndexColName].max() + 1

        print(f"Splitting cluster of size {len(sub_df)} into {num_sub_clusters} sub-clusters")
        new_clusters = getKMeansClusterIndexColumn(sub_df, num_sub_clusters, availableClusterIndex)
        df.loc[sub_df.index, clusterIndexColName] = new_clusters

        subsplit_count += 1

    # "normalize" the cluster values column
    df[clusterIndexColName] = pd.factorize(df[clusterIndexColName])[0]

    # # display histogram of the subgraph sizes
    # showHistogramOfSubgraphSizes(df[clusterIndexColName].value_counts())
    
    return df

def displaySizingInfoAboutDataset(df):
    print(f"Dataset size (# of rows): {len(df)}")
    print(f"dataframe columns: {df.columns.tolist()}")
    print()
    print("dataframe head:")
    print(df.head())
    print()

# will return a dataframe with 2 new columns added: artificialClusterIndex, artificialClusterGroupSize
def getArtificiallyClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                        disasterTypesOfInterest=["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],
                        dataClusteringBatchSize=6221, maxClusterSize=40, timeFeatureImportanceRatio=(1/6),
                        enableLoadingSavedOutputFile=True,
                        outputClusteredDatasetPath="Datasets/DisasterDeclarationSummariesPlusLocation_ArtificiallyClustered.csv"):
    
    # first try to read in the pre-saved clustered dataset (if exists)
    if enableLoadingSavedOutputFile and os.path.exists(outputClusteredDatasetPath):
        df = pd.read_csv(outputClusteredDatasetPath, index_col=None)
        displaySizingInfoAboutDataset(df)
        return df

    # retrieve the row/column-diltered dataset
    df = retrieveRowColFilteredDataset(inputDatasetPath = inputDatasetPath,
                                       disasterTypesOfInterest=disasterTypesOfInterest)
    
    clusteredBatches = []
    clusterIndexIncrement = 0
    for i in range(0, len(df), dataClusteringBatchSize):
        sub_df = df[i:min(len(df), i+dataClusteringBatchSize)]

        # normalize sub_df
        normalizer = DataNormalizer()
        sub_df = normalizer.normalize(sub_df, timeFeatureImportanceRatio=timeFeatureImportanceRatio)
        
        # cluster dataset, add columns "artificialClusterIndex" and "artificialClusterGroupSize" to dataset
        sub_df = getClusteredDataBatch(sub_df, max_cluster_size=maxClusterSize, clusterIndexColName="artificialClusterIndex")
        sub_df['artificialClusterGroupSize'] = sub_df["artificialClusterIndex"].map(sub_df["artificialClusterIndex"].value_counts())

        sub_df["artificialClusterIndex"] += clusterIndexIncrement
        clusterIndexIncrement = max(sub_df["artificialClusterIndex"]) + 1

        # denormalize sub_df and append it to clusteredBatches
        sub_df = normalizer.denormalize(sub_df)
        clusteredBatches.append(sub_df)
    print("Finished splitting all clusters!\n")
    
    # retain original dataframe, now with a nice "artificialClusterIndex" column added denoting the cluster each row belongs to
    df = pd.concat(clusteredBatches, axis=0)
    df["artificialClusterIndex"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "artificialClusterIndex")
    df["incidentId_categoricalEncoding"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentId_categoricalEncoding")
    
    df.to_csv(outputClusteredDatasetPath, index=False)
    displaySizingInfoAboutDataset(df)
    return df

# will return a dataframe with 2 new columns added: incidentIdClusterIndex, incidentIdClusterGroupSize
def getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                        disasterTypesOfInterest=["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],
                        dataClusteringMaxBatchSize=6221, maxClusterSize=40, timeFeatureImportanceRatio=(1/6),
                        enableLoadingSavedOutputFile=True,
                        outputClusteredDatasetPath="Datasets/DisasterDeclarationSummariesPlusLocation_IncidentIdClustered.csv"):
    
    # first try to read in the pre-saved clustered dataset (if exists)
    if enableLoadingSavedOutputFile and os.path.exists(outputClusteredDatasetPath):
        df = pd.read_csv(outputClusteredDatasetPath, index_col=None)
        displaySizingInfoAboutDataset(df)
        return df

    # retrieve the row/column-diltered dataset
    df = retrieveRowColFilteredDataset(inputDatasetPath = inputDatasetPath,
                                       disasterTypesOfInterest=disasterTypesOfInterest)
    
    clusteredBatches = []
    clusterIndexIncrement = 0
    uniqueIncidentIds = df["incidentId"].unique().tolist()

    for i in range(len(uniqueIncidentIds)):
        df_commonIncId = df[df["incidentId"]==uniqueIncidentIds[i]]
        for j in range(0, len(df_commonIncId), dataClusteringMaxBatchSize):
            sub_df = df_commonIncId[j:min(len(df_commonIncId), j+dataClusteringMaxBatchSize)]
            
            # normalize sub_df
            normalizer = DataNormalizer()
            sub_df = normalizer.normalize(sub_df, timeFeatureImportanceRatio=timeFeatureImportanceRatio)
        
            # cluster dataset, add columns "incidentIdClusterIndex" and "incidentIdClusterGroupSize" to dataset
            sub_df = getClusteredDataBatch(sub_df, max_cluster_size=maxClusterSize, clusterIndexColName="incidentIdClusterIndex")
            sub_df['incidentIdClusterGroupSize'] = sub_df["incidentIdClusterIndex"].map(sub_df["incidentIdClusterIndex"].value_counts())

            sub_df["incidentIdClusterIndex"] += clusterIndexIncrement
            clusterIndexIncrement = max(sub_df["incidentIdClusterIndex"]) + 1

            # denormalize sub_df and append it to clusteredBatches
            sub_df = normalizer.denormalize(sub_df)
            clusteredBatches.append(sub_df)
    print("Finished splitting all clusters!\n")
    
    # retain original dataframe, now with a nice "incidentIdClusterIndex" column added denoting the cluster each row belongs to
    df = pd.concat(clusteredBatches, axis=0)
    df["incidentIdClusterIndex"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentIdClusterIndex")
    df["incidentId_categoricalEncoding"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentId_categoricalEncoding")
    
    df.to_csv(outputClusteredDatasetPath, index=False)
    displaySizingInfoAboutDataset(df)
    return df




# driver code

def main():
    print()

    df = getIncidentIdClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    # df = getArtificiallyClusteredDataset(inputDatasetPath="Datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    print("Finished retrieving clustered dataset!\n")

    # maxClusterSize = max(df["artificialClusterGroupSize"])
    # print(f"Max cluster group size: {maxClusterSize}")
    
    print()

if __name__ == "__main__":
    main()

