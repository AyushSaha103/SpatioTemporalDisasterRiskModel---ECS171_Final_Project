import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os
from joblib import dump, load

def analyze_and_predict_disasters(file_path):
    output_dir = "disaster_pred_time_of_year_random_forest"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # assign region name based on latitude and longitude
    def get_region_name(lat, lon):
        ns = 'North' if lat > 40 else 'Central' if lat > 35 else 'South'
        ew = 'East' if lon > -98 else 'West'
        return f"{ns} {ew}"
    
    # prepare by creating new features
    def prepare_data_for_prediction(df):
        df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'])
        df['month'] = df['incidentBeginDate'].dt.month
        # season based on the month 
        df['season'] = pd.cut(df['incidentBeginDate'].dt.month, 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
        # region based on latitude and longitude
        df['region'] = df.apply(lambda row: get_region_name(row['Y'], row['X']), axis=1)
        
        X = pd.DataFrame({
            'latitude': df['Y'],
            'longitude': df['X'],
            'month': df['month'],
            # sine transformation 
            'month_sin': np.sin(2 * np.pi * df['month']/12),
             # cosine transformation
            'month_cos': np.cos(2 * np.pi * df['month']/12),
            'region': df['region']
        })

        # one-hot encode 'region' column
        X = pd.get_dummies(X, columns=['region'], prefix=['region'])
        
        return X

    # train the model for each disaster with cross-validation
    def train_disaster_models(df, disaster_types):
        X = prepare_data_for_prediction(df)
        print(X.columns)
        models = {}
        predictions = {}

        # StratifiedKFold ensures each fold has same class distribution
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        # hyperparameter grid for RandomizedSearchCV
        # n_estimators: # indivdual trees in the random forest
        # max_depth: max depth of each tree in the forest
        # max_features: limiting for amonth of features that each tree can consider
        # bootstrap: random sampling with replacement (should whole dataset or sample dataset be used)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt'],
            'bootstrap': [True, False]
        }

        for disaster in disaster_types:
            # target variable for current disaster type
            y = df[disaster]
            
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []

            #cross-validation to train and evaluate the model
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # RandomizedSearchCV for hyperparameters
                model = RandomForestClassifier(class_weight='balanced', random_state=42)
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1
                )
                
                random_search.fit(X_train, y_train)

                # choose best one from RandomizedSearchCV
                best_model = random_search.best_estimator_
                
                dump(best_model, f"ui_component/saved_prediction_models/rfc_disaster_models/random_forest_{disaster}.joblib")

                y_pred = best_model.predict(X_test)

                accuracy_scores.append(accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
                recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

            # average metrics after cross-validation
            print(f"\nMetrics for {disaster}:")
            print(f"  Accuracy: {np.mean(accuracy_scores):.2f}")
            print(f"  Precision: {np.mean(precision_scores):.2f}")
            print(f"  Recall: {np.mean(recall_scores):.2f}")
            print(f"  F1 Score: {np.mean(f1_scores):.2f}")

            # Store the trained model for this disaster
            models[disaster] = random_search.best_estimator_
            # Store the prediction probabilities
            predictions[disaster] = random_search.best_estimator_.predict_proba(X_test)[:, 1]

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nDone with {disaster}:")

        return models, predictions, X.columns

    # predict disaster risk by region for each month
    def predict_disasters_by_region(models, feature_columns):
        regions = ['North East', 'North West', 'Central East', 'Central West', 'South East', 'South West']
        months = range(1, 13)
        
        all_predictions = []
        
        for region in regions:
            if 'North' in region:
                lat = 42
            elif 'Central' in region:
                lat = 37
            else:
                lat = 32
                
            if 'East' in region:
                lon = -85
            else:
                lon = -110
            
            for month in months:
                feature_dict = {
                    'latitude': lat,
                    'longitude': lon,
                    'month': month,
                    'month_sin': np.sin(2 * np.pi * month/12),
                    'month_cos': np.cos(2 * np.pi * month/12)
                }
                
                for col in feature_columns:
                    if col.startswith('region_'):
                        feature_dict[col] = 1 if col == f'region_{region.replace(" ", "_")}' else 0
                
                features_df = pd.DataFrame([feature_dict])
                
                for col in feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                features_df = features_df[feature_columns]
                
                prediction_row = {
                    'region': region,
                    'month': month
                }
                
                # predict probabilities for each disaster type
                for disaster, model in models.items():
                    prob = model.predict_proba(features_df)[0][1]
                    prediction_row[disaster] = prob
                
                all_predictions.append(prediction_row)
        
        return pd.DataFrame(all_predictions)

    df = pd.read_csv(file_path)
    
    disaster_types = ['disasterType_storm', 'disasterType_flood', 'disasterType_rain',
                     'disasterType_ice', 'disasterType_snow', 'disasterType_blizzard',
                     'disasterType_hurricane']

    df['region'] = df.apply(lambda row: get_region_name(row['Y'], row['X']), axis=1)

    print("\nTraining prediction models...")
    models, predictions, feature_columns = train_disaster_models(df, disaster_types)
    regional_predictions = predict_disasters_by_region(models, feature_columns)
    
    regional_predictions['season'] = pd.cut(regional_predictions['month'], 
                                          bins=[0, 3, 6, 9, 12], 
                                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    plt.figure(figsize=(15, 10))
    for disaster in disaster_types:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=regional_predictions, x='region', y=disaster)
        plt.title(f'Average Probability of {disaster.replace("disasterType_", "")} by Region')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'regional_prediction_{disaster}.png'))
        plt.close()
    
    for disaster in disaster_types:
        plt.figure(figsize=(12, 8))
        seasonal_data = regional_predictions.pivot_table(
            index='region', 
            columns='season',
            values=disaster,
            aggfunc='mean'
        )
        sns.heatmap(seasonal_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title(f'Seasonal {disaster.replace("disasterType_", "")} Risk by Region')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'seasonal_regional_{disaster}.png'))
        plt.close()
    
    regional_predictions.to_csv(os.path.join(output_dir, 'regional_predictions.csv'), index=False)
    
    return models, regional_predictions

if __name__ == "__main__":
    try:
        models, predictions = analyze_and_predict_disasters('datasets/UpdatedDisasterDeclarationSummariesPlusLocation_RowColFiltered.csv')
    except Exception as e:
        print(f"Error: {e}")
