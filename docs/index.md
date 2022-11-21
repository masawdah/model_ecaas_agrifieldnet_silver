# ImbalancedCropDetectionML

{{ Model Description (paragraph) }}

![{{model_id}}](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `{{model_id}}`. Browse on [Radiant MLHub](https://mlhub.earth/model/{{model_id}}).

## Training Data

- [AgriFieldNet Competition Dataset - Source Imagery](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_source)
- [AgriFieldNet Competition Dataset - Test Labels](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_labels_train)



## Citation

{{

example:

Alasawedah, M. (2022) “A Spatio-Temporal Deep Learning-Based Crop Classification
Model for Satellite Imagery”, Version 1.0, Radiant MLHub.

}}

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
              [76.24483194693136,18.941440345823324],
              [76.24483194693136,28.326997605982278],
              [88.0460053723578,18.941440345823324],
              [88.0460053723578,28.326997605982278],
              [76.24483194693136,18.941440345823324]
             
          ]
        ]
      }
    }
  ]
}
```

## Applicable Temporal Extent

{{

The recommended start/end date of imagery for new inferencing. Example:

| Start | End |
|-------|-----|
| 2000-01-01 | present |

}}

## Learning Approach

{{

The learning approach used to train the model. It is recommended that you use
one of the values below, but other values are allowed.

- Supervised
- Unsupervised
- Semi-supervised
- Reinforcement-learning
- Other (explain)

}}

## Prediction Type

{{

The type of prediction that the model makes. It is recommended that you use one
of the values below, but other values are allowed.

- Object-detection
- Classification
- Segmentation
- Regression
- Other (explain)

}}

## Model Architecture

{{

Identifies the architecture employed by the model. This may include any string
identifiers, but publishers are encouraged to use well-known identifiers
whenever possible. More details than just “it’s a CNN”!

}}

## Training Operating System

{{

Identifies the operating system on which the model was trained.

- Linux
- Windows (win32)
- Windows (cygwin)
- MacOS (darwin)
- Other (explain)

}}

## Training Processor Type

{{

The type of processor used during training. Must be one of "cpu" or "gpu".

- cpu
- gpu

}}

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Methodology

{{

Use this section to provide more information to the reader about the model. Be
as descriptive as possible. The suggested sub-sections are as following:

}}

### Training

{{

Explain training steps such as augmentations and preprocessing used on image
before training.

}}

### Model

{{

Explain the model and why you chose the model in this section. A graphical representation
of the model architecture could be helpful to individuals or organizations who would
wish to replicate the workflow and reproduce the model results or to change the model
architecture and improve the results.

}}

### Structure of Output Data

{{

Explain output file names and formats, interpretation, classes, etc.

}}
