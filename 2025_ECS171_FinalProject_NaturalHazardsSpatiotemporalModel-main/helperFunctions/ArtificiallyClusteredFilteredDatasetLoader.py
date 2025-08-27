


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warning

from FilteredDatasetLoader import retrieveRowColFilteredDataset

# this script file is finished
# OUTPUT: filtered AND clustered dataframe, with these new columns:
#   -> "artificialClusterIndex", an integer index denoting the assigned spatiotemporal cluster group which each data row (node) belongs to
#   -> "artificialClusterGroupSize", an integer value denoting the size of each data row's respective cluster group
#   NOTE: the clustered groups have a max size of maxClusterSize nodes
# TO GET OUTPUT, call function: getClusteredDataset()

class DataNormalizer:
    def __init__(self):
        self.scaler_xy = StandardScaler()
        self.scaler_xyt = MinMaxScaler()
        self.timeUpscalingRatio = 1.0

    def normalize(self, df, timeFeatureImportanceRatio=1.0):
        df[["X", "Y"]] = self.scaler_xy.fit_transform(df[["X", "Y"]])
        df[["X", "Y", "timeInDays"]] = self.scaler_xyt.fit_transform(df[["X","Y","timeInDays"]])
        
        self.timeUpscalingRatio = 1 / timeFeatureImportanceRatio
        df["timeInDays"] *= self.timeUpscalingRatio
        return df
    
    def denormalize(self, df):
        df["timeInDays"] /= self.timeUpscalingRatio
        df[["X", "Y", "timeInDays"]] = self.scaler_xyt.inverse_transform(df[["X","Y","timeInDays"]])
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
    assignedClusterCol = kmeans.fit_predict(sub_df[['X', 'Y', "timeInDays"]])
    assignedClusterCol += availableClusterIndex
    return assignedClusterCol

def getClusteredDataBatch(df, max_cluster_size, method='ward', allowOversizedClusters=False):
    
    # run first stage of hierarchical clustering procedure
    data = df[['X', 'Y', "timeInDays"]].values
    Z = linkage(data, method=method)
    num_clusters = len(df) // max_cluster_size
    df["artificialClusterIndex"] = fcluster(Z, num_clusters, criterion='maxclust')
    cluster_sizes = df["artificialClusterIndex"].value_counts()

    if allowOversizedClusters:
        return df
    
    # split up oversized clusters into smaller ones
    subsplit_count = 0
    while cluster_sizes.max() > max_cluster_size:
        cluster_sizes = df['artificialClusterIndex'].value_counts()
        if cluster_sizes.max() <= max_cluster_size:
            break

        largest_cluster = cluster_sizes.idxmax()
        sub_df = df[df['artificialClusterIndex'] == largest_cluster]
        num_sub_clusters = int(len(sub_df) // max_cluster_size) + 1
        availableClusterIndex = df['artificialClusterIndex'].max() + 1

        print(f"Splitting cluster {largest_cluster}, size {len(sub_df)} into {num_sub_clusters} sub-clusters")
        new_clusters = getKMeansClusterIndexColumn(sub_df, num_sub_clusters, availableClusterIndex)
        df.loc[sub_df.index, 'artificialClusterIndex'] = new_clusters

        subsplit_count += 1

    # "normalize" the cluster values column
    df['artificialClusterIndex'] = pd.factorize(df['artificialClusterIndex'])[0]
    df['artificialClusterGroupSize'] = df['artificialClusterIndex'].map(df['artificialClusterIndex'].value_counts())

    # # display histogram of the subgraph sizes
    # showHistogramOfSubgraphSizes(df['artificialClusterIndex'].value_counts())
    
    return df

# will return a dataframe with 2 new columns added: artificialClusterIndex, artificialClusterGroupSize
def getClusteredDataset(inputDatasetPath="datasets/DisasterDeclarationsSummariesPlusLocation.csv",
                        enableLoadingSavedFile=False, dataClusteringBatchSize=4300, maxClusterSize=100, timeFeatureImportanceRatio=(1/6),
                        outputClusteredDatasetPath="datasets/DisasterDeclarationSummariesPlusLocation_ArtificiallyClustered.csv"):
    
    # first try to read in the pre-saved clustered dataset (if exists)
    if enableLoadingSavedFile and os.path.exists(outputClusteredDatasetPath):
        df = pd.read_csv(outputClusteredDatasetPath, index_col=None)
        return df

    # retrieve the row/column-diltered dataset
    df = retrieveRowColFilteredDataset(inputDatasetPath = inputDatasetPath)
    
    clusteredBatches = []
    clusterIndexIncrement = 0
    for i in range(0, len(df), dataClusteringBatchSize):
        sub_df = df[i:min(len(df), i+dataClusteringBatchSize)]

        # normalize sub_df
        normalizer = DataNormalizer()
        sub_df = normalizer.normalize(sub_df, timeFeatureImportanceRatio=timeFeatureImportanceRatio)
        
        # cluster dataset, add column "artificialClusterIndex" to dataset
        sub_df = getClusteredDataBatch(sub_df, max_cluster_size=maxClusterSize)
        sub_df["artificialClusterIndex"] += clusterIndexIncrement
        clusterIndexIncrement = max(sub_df["artificialClusterIndex"]) + 1

        # denormalize sub_df and append it to clusteredBatches
        sub_df = normalizer.denormalize(sub_df)
        clusteredBatches.append(sub_df)

    # retain original dataframe, now with a nice "artificialClusterIndex" column added denoting the cluster each row belongs to
    df = pd.concat(clusteredBatches, axis=0)
    df["artificialClusterIndex"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "artificialClusterIndex")
    df["incidentId_categoricalEncoding"], _ = getRandomlySortedValsOfCategoricallyEncodedColumn(df, "incidentId_categoricalEncoding")
    
    df.to_csv(outputClusteredDatasetPath, index=False)
    return df


# driver code

def main():
    print()

    df = getClusteredDataset(inputDatasetPath="datasets/DisasterDeclarationsSummariesPlusLocation.csv")
    print("Finished retrieving clustered dataset!\n")

    # maxClusterSize = max(df["artificialClusterGroupSize"])
    # print(f"Max cluster group size: {maxClusterSize}")

    print(f"Dataset size (# of rows): {len(df)}")
    print(f"dataframe columns: {df.columns.tolist()}")
    print()
    print("dataframe head:")
    print(df.head())

    print()

if __name__ == "__main__":
    main()

