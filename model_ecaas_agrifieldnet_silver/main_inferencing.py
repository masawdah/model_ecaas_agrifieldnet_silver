# Built-in modules
import json 

## Suppress Warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

# Basics of Python data handling 
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import shapely
import glob
import joblib
import argparse
from sklearn.preprocessing import QuantileTransformer
from utils import *

# Machine learning
from sklearn.metrics import log_loss
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("--INPUT_DATA", help="Path to input data", required=True)
parser.add_argument("--OUTPUT_DATA", help="Path to output data", required=True)

args = parser.parse_args()
INPUT_DATA = args.INPUT_DATA 
print(INPUT_DATA)
OUTPUT_DATA = args.OUTPUT_DATA

# Selected bands
selected_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
#print(f"{INPUT_DATA}/chips/Images/*")
imgs_folders = glob.glob(f"{INPUT_DATA}/chips/Images/*")  
#print(imgs_folders)
test_folder_ids = [i.split("/")[-1] for i in imgs_folders]
test_field_paths = [f'{INPUT_DATA}/chips/fields/{i}/field_ids.tif' for i in test_folder_ids]

#print(test_folder_ids)
#print(test_field_paths)

test_data = pd.DataFrame(test_folder_ids , columns=['unique_folder_id'])
test_data['field_paths'] = test_field_paths

extract_test_data = feature_extractor(test_data, INPUT_DATA)

# Each field has several pixels in| the data. 
# Compute the average values of the pixels within each field. 
# Use `groupby` to take the mean for each field_id

test_data_grouped  = extract_test_data.groupby(['field_id']).mean().reset_index()
test_data_grouped.field_id = [str(int(i)) for i in test_data_grouped.field_id.values]

test_df = test_data_grouped.copy()

# Features Engineering
test_veg_indices = veg_indices(test_df, data_type='test')
test_rededge_indices = rededge_indices(test_df, data_type='test')
test_bloom_indices = bloom_indices(test_df, data_type='test')


# Merge raw Sentinel-2 data with new extracted features
test_df = pd.merge(test_df, test_veg_indices, on=['field_id'], how='inner')
test_df = pd.merge(test_df, test_rededge_indices, on=['field_id'], how='inner')
test_df = pd.merge(test_df, test_bloom_indices, on=['field_id'], how='inner')

test_df.to_csv(f'{OUTPUT_DATA}/veg_indices.csv', index=False)
test_df = pd.read_csv(f'{OUTPUT_DATA}/veg_indices.csv')

# Spatial variability for spectral & indices data
## Extract fields centroid coordinates
te_centroid_fields, te_lats, te_longs = fields_centroids(test_data, INPUT_DATA)
te_gdf = pd.DataFrame({'field_id' : te_centroid_fields, 'lat' : te_lats,'long' : te_longs })

## columns
cols = list(test_df.columns)
cols.remove('field_id')

## Convert dataframes to geoataframe to perform the zonal statistcs 
te_gdf = pd.merge(te_gdf, test_df,  on=['field_id'], how='inner')
te_gdf = gpd.GeoDataFrame(te_gdf, geometry=gpd.points_from_xy(te_gdf.long, te_gdf.lat))

## Spatial variabilty within different radiuses
keep_same = {'field_id'}
buffer_500m_stats = spatial_variability(te_gdf, 500, cols)
buffer_500m_stats = add_suffix(buffer_500m_stats, '_500m', keep_same)


buffer_1000m_stats = spatial_variability(te_gdf, 1000, cols)
buffer_1000m_stats = add_suffix(buffer_1000m_stats, '_1000m', keep_same)

buffer_1500m_stats = spatial_variability(te_gdf, 1500, cols)
buffer_1500m_stats = add_suffix(buffer_1500m_stats, '_1500m', keep_same)

buffer_2500m_stats = spatial_variability(te_gdf, 2500, cols)
buffer_2500m_stats = add_suffix(buffer_2500m_stats, '_2500m', keep_same)

buffer_3500m_stats = spatial_variability(te_gdf, 3500, cols)
buffer_3500m_stats = add_suffix(buffer_3500m_stats, '_3500m', keep_same)

buffer_5000m_stats = spatial_variability(te_gdf, 5000, cols)
buffer_5000m_stats = add_suffix(buffer_5000m_stats, '_5000m', keep_same)

# More field stats
test_df = field_stats(test_df, extract_test_data)
test_df.to_csv(f"{OUTPUT_DATA}/field_stats_indices.csv", index=False)

# Merge everything togther
test_df = pd.read_csv(f"{OUTPUT_DATA}/field_stats_indices.csv")

test_df = pd.merge(test_df, buffer_500m_stats, on=['field_id'], how='inner')
test_df = pd.merge(test_df, buffer_1000m_stats, on=['field_id'], how='inner')
test_df = pd.merge(test_df, buffer_1500m_stats, on=['field_id'], how='inner')
test_df = pd.merge(test_df, buffer_2500m_stats, on=['field_id'], how='inner')
test_df = pd.merge(test_df, buffer_3500m_stats, on=['field_id'], how='inner')
test_df = pd.merge(test_df, buffer_5000m_stats, on=['field_id'], how='inner')

# Fill missing data
test_df = test_df.fillna(-999999).replace(np.inf, -999999).replace(-np.inf, -999999)

# Apply QuantileTransformer to have normal distribution
X_test = test_df.drop(columns=['field_id'])
qt=QuantileTransformer(output_distribution="normal",random_state=2022)
X_test= pd.DataFrame(qt.fit_transform(X_test),columns=X_test.columns)

# Prediction using catboost
catboostpreds= []
for i in range(10):
    cat_model = joblib.load(f"{INPUT_DATA}/checkpoint/cats/cat{i+1}.sav")
    catboostpred = cat_model.predict_proba(X_test)
    catboostpreds.append(catboostpred)

# Prediction using lgbm
lgbmpreds= []
for i in range(10):
    lgbm_model = joblib.load(f"{INPUT_DATA}/checkpoint/lgbms/lgbm{i+1}.sav")
    lgbmpred = lgbm_model.predict_proba(X_test)
    lgbmpreds.append(lgbmpred)


# Prediction using lgbm
xgbmpreds= []
dtest = xgb.DMatrix(X_test)
for i in range(10):
    xgbm_model = joblib.load(f"{INPUT_DATA}/checkpoint/xgbms/xgbm{i+1}.sav")
    #xgbmpred = xgbm_model.predict_proba(X_test)
    xgbmpred = xgbm_model.predict(dtest, iteration_range=(0, xgbm_model.best_iteration + 1))
    xgbmpreds.append(xgbmpred)
    
# Ensemble predictions
lgbmpreds_mean = np.mean(lgbmpreds, axis=0)
cbpreds_mean = np.mean(catboostpreds, axis=0)
xgbmpreds_mean = np.mean(xgbmpreds, axis=0)

# weighted average models
preds = 0.70*(0.50*lgbmpreds_mean + 0.50*xgbmpreds_mean) + 0.30* cbpreds_mean

# Put the predictions into dataframe
predictions = pd.DataFrame(preds)
predictions['field_id'] = list(test_df['field_id'])

predictions = predictions.rename(columns={
    'field_id':'field_id',
    0:'Wheat',
    1:'Mustard', 
    2:'Lentil',
    3:'No Crop',
    4:'Sugarcane',
    5:'Garlic',
    6:'Potato',
    7:'Green pea',
    8:'Bersem',
    9:'Coriander',
    10:'Gram',
    11:'Maize',
    12:'Rice'
})

cols = list(predictions.columns)
cols.remove("field_id")
cols1 = ['field_id']
cols1.extend(cols)
predictions = predictions[cols1]

predictions.to_csv(f'{OUTPUT_DATA}/predictions.csv', index=False)


                            

