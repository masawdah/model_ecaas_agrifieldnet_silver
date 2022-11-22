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


# Selected bands
selected_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']


#Create an X variable. 
#Each row is a pixel and each column is one of the band observations mapped to its corresponding field.
def feature_extractor(data_):
    '''
        data_: Dataframe with 'field_paths' and 'unique_folder_id' columns
        path: Path to source collections files

        returns: pixel dataframe with corresponding field_ids
        '''
    
    X = np.empty((0, n_selected_bands * n_obs))
    X_tile = np.empty((img_sh * img_sh, 0))
    X_arrays = []
        
    field_ids = np.empty((0, 1))

    for idx, tile_id in tqdm(enumerate(data_['unique_folder_id'])):
        
        field_src =   rasterio.open( data_['field_paths'].values[idx])
        field_array = field_src.read(1)
        field_ids = np.append(field_ids, field_array.flatten())
        
        bands_src = [rasterio.open(f"{INPUT_DATA}/chips/Images/{tile_id}/{band}.tif") for band in selected_bands]
        bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]
        
        X_tile = np.hstack(bands_array)

        X_arrays.append(X_tile)
        

    X = np.concatenate(X_arrays)
    
    data = pd.DataFrame(X, columns=selected_bands)

    data['field_id'] = field_ids

    return data[data['field_id']!=0]


# Extract fields centroids coordinates 
def fields_centroids(data_):
    '''
        data_: Dataframe with 'field_paths' and 'unique_folder_id' columns
        path: Path to source collections files

        returns: fileds dataframe with centroids coordinates
        '''
    
    longs = []
    lats = []
    centroid_fields = []
    for idx, tile_id in tqdm(enumerate(data_['unique_folder_id'])):
        field_src =   rasterio.open( data_['field_paths'].values[idx])
        field_array = field_src.read(1)
        fields = list(np.unique(field_array))
        
        band = rasterio.open(f"{INPUT_DATA}/chips/Images/{tile_id}/B01.tif")
        band_array = band.read(1)

        for field in fields:
            if field !=0:
                if field not in centroid_fields:
                    centroid_fields.append(field)
                    mask = field_array == field
                    polys = []
                    for s ,v in shapes(band_array, mask=mask, transform=band.transform):
                        polys.append(shapely.geometry.shape(s))
                    poly = shapely.ops.unary_union(polys)
                    centroid = poly.centroid
                    long = centroid.x
                    longs.append(long)
                    lat = centroid.y
                    lats.append(lat)
                    
    return centroid_fields, lats, longs


def veg_indices(df, data_type='train'):
    veg_df = pd.DataFrame()
    
    veg_df['MNDVI'] = (df['B08'] - df['B04'] )  / (df['B08'] +df['B04'] - 2*df['B02'])
    veg_df['NDSI'] =  df['B03']  / (df['B11'])
    veg_df['NDVI'] =  (df['B08'] - df['B04'] )  / (df['B08'] +df['B04'] )
    veg_df['NDWI'] = (df['B03'] -  df['B08'] )  / (df['B03'] +df['B08']) 
    veg_df['NDMI'] = (df['B08'] - df['B11'] )  / (df['B08'] +df['B11' ])
    veg_df['NDBI'] = (df['B11'] - df['B08'] )  / (df['B11'] +df['B08'])
    veg_df['NDCI'] = (df['B05'] - df['B04'] )  / (df['B05'] +df['B04'])
    veg_df['GNDVI'] = (df['B08'] - df['B03'] )  / (df['B08'] +df['B03'] )
    veg_df['SAVI'] = (df['B01'] - df['B02'] )  / (df['B01'] +df['B02'] + 0.428 ) * (1.0 + 0.428)
    veg_df['EVI2'] = 2.4*((df['B01'] - df['B02'] )  / (df['B01'] +df['B02'] + 1.0))
    veg_df['EVI'] = 2.5*((df['B04'] - df['B03'] )  / (df['B04'] + 6 * df['B03'] - 7.5 * df['B01'] + 1.0))
    veg_df['BSI'] =  (df['B11'] - df['B04'] )  / (df['B08'] +df['B02'])
    veg_df['NDVI_R'] =  (df['B08'] - df['B07'] )  / (df['B08'] +df['B07'])
    veg_df['CHL'] =  (df['B07'] / (df['B05']))  - 1
    veg_df['CVI'] =  (df['B08'] / (df['B03']))  * (df['B04'] / (df['B03']))
    veg_df['BI'] =  (df['B04'] **2+ df['B03']**2+ df['B02']*2) /3
    veg_df['SI'] =  (df['B04'] - df['B02'])  / (df['B04'] + df['B02'])

    veg_df['NMDI'] =  (df['B08'] - (df['B11'] - df['B12']))  / (df['B08'] + (df['B11'] - df['B12']))
    veg_df['WBI'] =  df['B8A']  / (df['B09'])
    veg_df['MSI'] =  df['B11']  / (df['B08'])
    veg_df['BSI1'] =  ((df['B12'] + df['B04']) - (df['B8A'] + df['B02']))  / ((df['B12'] + df['B04']) + (df['B8A'] + df['B02']))
    veg_df['BSI2'] =  ((df['B11'] + df['B04']) - (df['B8A'] + df['B02']))  / ((df['B11'] + df['B04']) + (df['B8A'] + df['B02']))
    veg_df['BSI3'] =  100 * np.sqrt((df['B12'] - df['B03']) / (df['B12'] + df['B03']))
    veg_df['NDSI1'] = (df['B11'] - df['B8A'] )  / (df['B11'] + df['B8A'] )
    veg_df['NDSI2'] = (df['B12'] - df['B03'] )  / (df['B12'] + df['B03'] )
    veg_df['BI2'] = df['B04'] + df['B11'] - df['B8A']
    veg_df['DBSI'] = veg_df['NDSI2'] - ((df['B8A'] - df['B04']) / (df['B8A'] + df['B04']))
    veg_df['MBI'] = ((df['B11'] + df['B12'] + df['B8A']) / (df['B11'] + df['B12'] + df['B8A'])) + 0.5
    
    veg_df['R01'] =  df['B01']  / (df['B03'])
    veg_df['R02'] =  df['B01']  / (df['B05'])
    veg_df['R03'] =  df['B11']  / (df['B12'])
    veg_df['R04'] =  df['B05']  / (df['B04'])

    veg_df['MI'] = (df['B8A'] - df['B11'] )  / (df['B8A'] + df['B11'] )
    veg_df['MRESR'] = (df['B06'] - df['B01'] )  / (df['B05'] - df['B01'])
    veg_df['PSRI'] = (df['B04'] - df['B02'] )  / (df['B06'] )
   
    veg_df['TVI'] = (120*(df['B06'] - df['B03'] ) - 200 * (df['B04'] - df['B03'])) / 2
    
    
    veg_df['ARVI'] = (df['B08'] - 2*df['B04'] + df['B02'])  / (df['B08'] + 2*df['B04'] + df['B02'])

    veg_df['SIPI'] =  (df['B08'] - df['B02'] )  / (df['B08'] +df['B04'] )

    veg_df['EXG'] = (2 * df['B03'] -  df['B04'] -  df['B02'] )
    veg_df['ACI'] = (df['B08']  )  * (df['B04'] +df['B03'] )
    
    
    veg_df['field_id'] = list(df['field_id'])
    
    if data_type == 'train':
        veg_df['crop_id'] = list(df['crop_id'])
    return veg_df



def rededge_indices(df, data_type='train'):
    rededge_df = pd.DataFrame()
    # Redge Edge Indices
    rededge_df['NDVIre1'] =  (df['B08'] - df['B05'])  / (df['B08'] + df['B05'])
    rededge_df['NDVIre2'] =  (df['B08'] - df['B06'])  / (df['B08'] + df['B06'])
    rededge_df['NDVIre3'] =  (df['B08'] - df['B07'])  / (df['B08'] + df['B07'])

    rededge_df['NDRE1'] =  (df['B06'] - df['B05'])  / (df['B06'] + df['B05'])
    rededge_df['NDRE2'] =  (df['B07'] - df['B05'])  / (df['B07'] + df['B05'])
    rededge_df['NDRE3'] =  (df['B07'] - df['B06'])  / (df['B07'] + df['B06'])

    rededge_df['CIre1'] =  (df['B08'] /(df['B05']))  - 1 
    rededge_df['CIre2'] =  (df['B08'] /(df['B06']))  - 1
    rededge_df['CIre3'] =  (df['B08'] /(df['B07']))  - 1

    rededge_df['MCARI1'] =  ((df['B05'] - df['B04']) - 0.2*(df['B05'] - df['B03'])) * (df['B05'] / (df['B04']))
    rededge_df['MCARI2'] =  ((df['B06'] - df['B04']) - 0.2*(df['B06'] - df['B03'])) * (df['B06'] / (df['B04']))
    rededge_df['MCARI3'] =  ((df['B07'] - df['B04']) - 0.2*(df['B07'] - df['B03'])) * (df['B07'] / (df['B04']))

    
    rededge_df['TCARI1'] =  3*((df['B05'] - df['B04']) - 0.2*(df['B05'] - df['B03'])) * (df['B05'] / (df['B04']))
    rededge_df['TCARI2'] =  3*((df['B06'] - df['B04']) - 0.2*(df['B06'] - df['B03'])) * (df['B06'] / (df['B04']))
    rededge_df['TCARI3'] =  3*((df['B07'] - df['B04']) - 0.2*(df['B07'] - df['B03'])) * (df['B07'] / (df['B04']))

    rededge_df['MTCI1'] =  (df['B06'] - df['B05'])  / (df['B05'] - df['B04'])
    rededge_df['MTCI2'] =  (df['B07'] - df['B05'])  / (df['B05'] - df['B04'])
    rededge_df['MTCI3'] =  (df['B07'] - df['B06'])  / (df['B06'] - df['B04'])  
    
    rededge_df['field_id'] = list(df['field_id'])
    if data_type == 'train':
        rededge_df['crop_id'] = list(df['crop_id'])
    return rededge_df



def bloom_indices(df, data_type='train'):
    bloom_df = pd.DataFrame()
    # Blooming Indices (to detect flowers colors (purple, yellow) of different crops)
    bloom_df['NDGI'] =  (df['B04'] - df['B03'] )  / (df['B04'] +df['B03'] )
    bloom_df['DYI'] =  df['B04']  / df['B03']
    bloom_df['NDPI'] =  (0.5*(df['B04'] + df['B02']) - df['B03'])  / (0.5*(df['B04'] + df['B02']) + df['B03'])
    bloom_df['PEBI'] =  bloom_df['NDPI'] / ((bloom_df['NDGI'] +1) * df['B08'])
    bloom_df['NDYI'] =  (0.5*(df['B04'] + df['B03']) - df['B02'])  / (0.5*(df['B04'] + df['B03']) + df['B02'])
    bloom_df['YEBI'] =  bloom_df['NDYI'] / ((bloom_df['NDGI'] +1) * df['B08']) 
    
    bloom_df['field_id'] = list(df['field_id'])
    if data_type == 'train':
        bloom_df['crop_id'] = list(df['crop_id'])
    return bloom_df

def spatial_variability(gdf, radius, cols):
    '''
        gdf: GeoDataframe with all extracted features and 'field_id' columns
        radius: radius
        cols: columns names of extracted features

        returns: dataframe with statistcs within specified radius around each filed
        '''
    gdf_buffer = gdf.buffer(radius)
    buffer_stats = None
    for idx, site_buffer in tqdm(enumerate(gdf_buffer)):
        a = gdf.within(site_buffer)
        b = gdf[a].reset_index(drop=True)
        stats_pt = b[cols].agg([np.nanmean , np.nanmin, np.nanmax, np.nanstd])
        stats_pt.index = stats_pt.index
        stats_pt_out = stats_pt.stack()
        stats_pt_out.index = stats_pt_out.index.map('{0[1]}_{0[0]}'.format)
        stats_pt_out = stats_pt_out.to_frame().T

        if buffer_stats is None:
            buffer_stats = stats_pt_out
        else:
            buffer_stats = pd.concat([buffer_stats, stats_pt_out],ignore_index=True)
            
    buffer_stats['field_id'] = list(gdf['field_id'])
    
    return buffer_stats


def field_stats(te_df, test_data):
    test_data_grouped = test_data.groupby(['field_id']).median().reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"median_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).std().reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"std_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).min().reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"min_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)


    test_data_grouped = test_data.groupby(['field_id']).max().reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"max_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).sum().reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"sum_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).quantile(0.75).reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"q75_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).quantile(0.10).reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"q10_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)

    test_data_grouped = test_data.groupby(['field_id']).quantile(0.25).reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"q25_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)


    test_data_grouped = test_data.groupby(['field_id']).agg(lambda x:x.value_counts().index[-1]).reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"minority_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)


    test_data_grouped = test_data.groupby(['field_id']).agg(lambda x:x.value_counts().index[0]).reset_index()
    test_data_grouped = test_data_grouped.drop(columns=['field_id'])
    for col in selected_bands:
        test_data_grouped = test_data_grouped.rename(columns={f"{col}":f"majority_{col}"})
    te_df= pd.concat([te_df, test_data_grouped], axis=1)
    
    return te_df


def add_suffix(df, suffix , keep_same):
    df.columns = ['{}{}'.format(c, '' if c in keep_same else suffix) 
                  for c in df.columns]
    return df

