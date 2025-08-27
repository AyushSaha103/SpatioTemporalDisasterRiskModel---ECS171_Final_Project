
Last updated: 2025-02-24


Please note we have removed our .csv files in order for the size to not exceed 50 MB as instructions mentioend because it was 90 MB if we had kept it in. To see these CSV's , please visit our github repo on github : https://github.com/AyushSaha103/2025_ECS171_FinalProject_NaturalHazardsSpatiotemporalModel

# General Code Base Functionalities Provided

Visualizers In Main Branch:
- ### DisasterVisualizer.py
	- Run this script to retrieve the dataset and view two 3D interactive plots, each showing all the disaster occurences from the dataset.
	- The plot axes are X (latitude), Y (longitude), and *timeOfStartInDays* (number of days since *1953-05-15*).
	- The plot has its points color-coded in a certain (user-defined) manner, i.e. by the points' categorically encoded incidentId's or their incidentId Cluster Group Assignment Index.

- ### DisasterDatasetTimelapseAnimator.py
	- Run this standalone script to view a timelapse animation of the disasters in our dataset occuring on a 2D grid, whose axes are X (latitude) and Y (longitude).
		- The disasters are color-coded by their incident ID.

- ### lstm_demo.py
	- Run this script to view an example prediction on a 1 year time frame at specific locations.  Also displays ground truth data for comparison

 - ### To Run the Random Forest Classification for Disaster Prediction:
   	- Navigate to Main
   	- Go to the disaster_pred_time_of_year_random_forest folder
   	- Navigate to the README.md file inside the folder to get more background around what the RFC algorithm will do for this dataset
   	- Press play on the disasterpred_random_forest.py file to see the algorithm run
   	- The charts and diagrams will populate inside the disaster_pred_time_of_year_random_forest folder

- ### To run the Funding Prediction Model
  	- Navigate to PredictionModels/Funding_Prediction
  	- Run FundingPrediction.ipynb to view the EDA, model training, and model evaluation for the funding model.
  	- Run Funding_Visualizer.py to generate a visualization of funding categorizations over the dataset.
  	  
- ### Web UI Demo Component
	- Ensure that streamlit is downloaded on your device (use the command below if not)
   		``` pip install streamlit ```
	- To run the web UI component, navigate to the ui_component directory
		```
		cd ui_component
		```
	- Run the following command to save models in an organized manner
		```
		mkdir saved_prediction_models
		cd saved_prediction_models
		mkdir rfc_disaster_models
		cd ../..
		```
	- Enter the command below
 		```
		streamlit run app.py
		```
   		- NOTE: If the 'saved_prediction_models' directory does not exist, make sure to train all the models before running the UI to ensure the models have been saved
	- To see this running locally, either double-click on the localhost link OR
	- Navigate to your browser and enter in http://localhost:8501

# Specific Code Base Functionalities Provided

- ### DatasetLoaders/ ClusteredFilteredDatasetLoader.py
	- function of interest:

		```
		getIncidentIdClusteredDataset(

			inputDatasetPath="",

			disasterTypesOfInterest=["storm", "flood", "rain", "ice", "snow", "blizzard", "hurricane"],

			dataClusteringMaxBatchSize=6221, maxClusterSize=40, timeFeatureImportanceRatio=(1/6),

			enableLoadingSavedOutputFile=True,
			outputClusteredDatasetPath=""

		) -> pd.DataFrame
		```
	- function of interest parameters:

		- inputDatasetPath: use "Datasets/DisasterDeclarationsSummariesPlusLocation.csv"

		- disasterTypesOfInterest: list of disaster types to include in retrieved dataframe
			- must be a subset of this list: ["fire", "tornado", "tsunami", "snow", "ice", "rain", "landslide", "flood", "blizzard", "wind", "thunderstorm", "storm", "earthquake", "hurricane", "temperature", "freez"]

		- dataClusteringMaxBatchSize: (int) maximum permissable size of each data batch which the clustering algorithm operates on at once
		- maxClusterSize: (int) maximum allowed assigned cluster group size
		- timeFeatureImportanceRatio: (float) clustering algorithm's perceived importance of time over spatial position
			- set to 1.0 to equalize the importance measure
			- increase parameter to increase the relative importance of time
			- decrease parameter to decrease the relative importance of time

		- enableLoadingSavedOutputFile: (boolean) set to True if you want to enable caching of the clustered dataset
			- note: if it's true, the function will not update the saved cached dataset, even if the other function parameters are changed
		- outputClusteredDatasetPath: (string) file path for clustering algorithm to save/cache its output dataset
	
	- function of interest return value (pd.DataFrame):
		- semantic meaning: *the (cleaned/ filtered/ clustered) dataset*
		- details:
			- size: 49768 rows
			- columns: ['incidentBeginDate', 'incidentEndDate', 'X', 'Y', 'declarationTitle', 'incidentId', 'incidentId_categoricalEncoding', 'commonIncidentIdGroupSize', 'timeOfStartInDays', 'timeOfEndInDays', 'incidentDurationInDays', 'disasterType_storm', 'disasterType_flood', 'disasterType_rain', 'disasterType_ice', 'disasterType_snow', 'disasterType_blizzard', 'disasterType_hurricane', 'disasterType_categoricalEncoding', 'incidentIdClusterIndex', 'incidentIdClusterGroupSize']

				- 'incidentBeginDate': (string) incident's start date (i.e. "2005-08-27")
				- 'incidentEndDate': (string) incident's end date (i.e. "2005-10-01")
				- 'X': (float) incident's latitude coordinate
				- 'Y': (float) incident's longitude coordinate
				- 'declarationTitle': (string) incident's declared title
				- 'incidentId': (integer) raw identifier for incident assigned by FEMA
				- 'incidentId_categoricalEncoding': (integer) updated identifier for incident--simply a mapping of 'incidentID' to linearly increasing values starting from 0
				- 'commonIncidentIdGroupSize': (integer) size of the group of all incidents in the dataset sharing the same incident ID
				- 'timeOfStartInDays': (integer) number of days between 1953-05-15 and incident's start date
				- 'timeOfEndInDays': (integer) number of days between 1953-05-15 and incident's end date
				- 'incidentDurationInDays': (integer) number of days for which incident has lasted
				- 'disasterType_<some type>': (binary integer) one-hot encoded value denoting the incident type
				- 'disasterType_categoricalEncoding': (integer) unique identifier for the incident's one-hot encoded disaster type. (This is needed because some incidents have multiple disaster types).
				- 'incidentIdClusterIndex': (integer) index of cluster group to which incident was assigned
					- note: the cluster groups were assigned by treating seperate incident ID groups as their own clusters, and breaking apart large cluster groups until all groups had size <= maxClusterSize
				- 'incidentIdClusterGroupSize': (integer) size of the assigned cluster group to which incident belongs

- ### DatasetVisualization/ DisasterDatasetVisualizers.py
	- function of interest:

 		**```generateInteractive_3D_NodePlot(df, colorShadedColumn="commonIncidentIdGroupSize")```**
	- function of interest parameters:

		- df: (pd.DataFrame) the clustered disaster declarations dataframe
		- colorShadedColumn: (string) the column of the dataframe whose values to use in color-coding the plotted points

- ### DisasterDatasetTimelapseAnimator.py
	- function of interest:
		- N/A; the entire standalone script provides a standalone animation functionality
		- note: there should not be any modules inheriting anything from this script
	- parameter of interest:
		```SLIDING_TIME_WINDOW_DURATION_IN_YEARS = 9```

		- this parameter controls the duration of the time window visualized in each frame of the animation

### To run demo with visuals run demo.py
	- This file displays a graph with points representing disasters as well as no-disaster points for a specific time range, e.g., 20000 - 20365
	- It then takes these points and plugs them into the lstm model to predict the probability of a disaster at all these points and graphs these onto another scatter plot.
