# Weighted Tree-based Crop Classification Models for Imbalanced datasets

Second place solution to classify crop types in agricultural fields across Northern India using multispectral observations from Sentinel-2 satellite. Ensembled weighted tree-based models "LGBM, CATBOOST, XGBOOST" with stratified k-fold cross validation, taking advantage of spatial variabilty around each field within different distances.

![model_ecaas_agrifieldnet_silver_v1](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `model_ecaas_agrifieldnet_silver_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_silver_v1).

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|30 GB RAM | 30 GB RAM|


## Get Started With Inferencing

First clone this Git repository.

```bash
git clone https://github.com/masawdah/model_ecaas_agrifieldnet_silver.git
cd model_ecaas_agrifieldnet_silver/
```

After cloning the model repository, you can use the Docker Compose runtime
files as described below.

## Pull or Build the Docker Image

{{

:pushpin: Model developer: please build and publish your images to [Docker
Hub](https://hub.docker.com/). The images should be public, and should be
tagged as `model_id:version` and `model_id:version-gpu`.

For example model_id `model_unet_agri_western_cape_v1`
would have two docker image tags published on Docker Hub:

* `model_unet_agri_western_cape:1` for cpu inferencing
* `model_unet_agri_western_cape:1-gpu` for gpu inferencing

}}

Pull pre-built image from Docker Hub (recommended):

```bash
# cpu
docker pull docker.io/{{your_org_name}}/{{repository_name}}:1
# optional, for NVIDIA gpu
docker pull docker.io/{{your_org_name}}/{{repository_name}}:1-gpu

```

Or build image from source:

```bash
# cpu
docker build -t {{your_org_name}}/{{repository_name}}:1 -f Dockerfile_cpu .
# optional, for NVIDIA gpu
docker build -t {{your_org_name}}/{{repository_name}}:1-gpu -f Dockerfile_gpu .

```

## Run Model to Generate New Inferences

1. Prepare your input and output data folders. The `data/` folder in this repository
    contains some placeholder files to guide you.

    * The `data/` folder must contain:
        * `input/chips` Sentinel-2 10m imagery chips  for inferencing:
            * `Images` Sentinel-2 10m imagery chips for inferencing:
                * Folder name `chip_id` e.g. `00c23`  Sentinel-2 bands 10m:
                     * File name: `B01.tif` Type=Byte, ColorInterp=Coastal
                     * File name: `B02.tif` Type=Byte, ColorInterp=Blue
                     * File name: `B03.tif` Type=Byte, ColorInterp=Green
                     * File name: `B04.tif` Type=Byte, ColorInterp=Red
                     * File name: `B05.tif` Type=Byte, ColorInterp=RedEdge
                     * File name: `B06.tif` Type=Byte, ColorInterp=RedEdge
                     * File name: `B07.tif` Type=Byte, ColorInterp=RedEdge
                     * File name: `B08.tif` Type=Byte, ColorInterp=NIR
                     * File name: `B8A.tif` Type=Byte, ColorInterp=NIR08
                     * File name: `B09.tif` Type=Byte, ColorInterp=NIR09
                     * File name: `B11.tif` Type=Byte, ColorInterp=SWIR16
                     * File name: `B12.tif` Type=Byte, ColorInterp=SWIR22
                   
                     * File Format: GeoTIFF, 256x256
                     * Coordinate Reference System: WGS84 / UTM
            * `fields` Corresponding field ids for each pixel in Sentinel-2 images:
                * Folder name: `chip_id` e.g. `00c23`  Corresponding field ids:
                     * File name: `field_ids.tif`
           
                     * File Format: GeoTIFF, 256x256
                     * Coordinate Reference System:  WGS84 / UTM
             
        * `/input/checkpoint` the model checkpoint {{ file | folder }}, `{{ checkpoint file or folder name }}`.
            Please note: the model checkpoint is included in this repository.
    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_silver/data/input"
    export OUTPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_silver/data/output"
    ```

3. Run the appropriate Docker Compose command for your system

    ```bash
    # cpu
    docker compose up {{model_id}}_cpu
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
