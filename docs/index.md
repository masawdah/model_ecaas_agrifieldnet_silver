# Weighted Tree-based Crop Classification Models for Imbalanced Datasets

Second place solution to classify crop types in agricultural fields across Northern India using multispectral observations from Sentinel-2 satellite. Ensembled weighted tree-based models "LGBM, CATBOOST, XGBOOST" with stratified k-fold cross validation, taking advantage of spatial variabilty around each field within different distances.

![model_ecaas_agrifieldnet_silver_v1](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `model_ecaas_agrifieldnet_silver_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_silver_v1).

## Training Data

- [AgriFieldNet Competition Dataset - Source Imagery](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_source)
- [AgriFieldNet Competition Dataset - Test Labels](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_labels_train)


## Citation

Alasawedah, M. (2022) “A Spatio-Temporal Deep Learning-Based Crop Classification
Model for Satellite Imagery”, Version 1.0, Radiant MLHub.

## License

CC-BY-4.0

## Creator{{s}}

Mohammad Alasawedah - Earth Observation and Climate Data Science
[https://www.linkedin.com/in/mohammad-alasawdah-b3b541a5/](https://www.linkedin.com/in/mohammad-alasawdah-b3b541a5/)


## Contact

masawdah@gmail.com

## Applicable Spatial Extent

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [76,18],
              [76,28],
              [88,18],
              [88,28],
              [76,18]
          ]
        ]
      }
    }
  ]
}
```

## Applicable Temporal Extent

| Start | End |
|-------|-----|
| 2022-01-01 | 2022-05-31 |


## Learning Approach

- Supervised


## Prediction Type

- Classification


## Training Operating System

- Linux

## Training Processor Type

- cpu

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Methodology


### Training

Prepare the data for tree models by computing the average values of the pixels within each field, then feature engineering by computing spatial variabilty, more vegetation, and flowering phenology indices.

Zonal statistcs (mean , min, max, std) within different radiuses (0.50, 1.00, 1.50, 2.50, 3.50, 5.00) Km around each field

### Model

Wighted average tree-based models: lightgbm. catboost, xgboost classifers.

### Structure of Output Data
- Predictions.csv: Final predictions text file, with 13 crops classes as following `Wheat, Mustard, Lentil, No Crop, Sugarcane, Garlic, Potato, Green pea, Bersem, Coriander, Gram, Maize, Rice` 

- veg_indices.csv: Extracted vegitation indices for each field.

- Field_stats_indices.csv: Extracted statisitcs for each field. 

