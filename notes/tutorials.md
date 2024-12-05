---
title: Tutorials
subject: Tutorials
short_title: Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CSIC
      - UCM
      - IGEO
    orcid: 0000-0002-6739-0053
    email: juanjohn@ucm.es
license: CC-BY-4.0
keywords: notation
---


## Datasets

* AEMET
* E-OBS
* ERA5

---
## Extremes


---
## Time Series


---
## Spatial Fields


---
## Spatiotemporal Fields


---
## Running Experiments


### I: Data

First, we need to download some data.

**AEMET**

```bash
dvc repro pipelines/download/dvc.yaml
dvc repro pipelines/download/dvc.yaml:download_aemet
dvc repro pipelines/download/dvc.yaml:download_gmst
dvc exp run pipelines/viz/dvc.yaml:viz_station_t2max --set-param eda.station_id='3129A' --downstream --force
```

**GMST**

```bash
dvc repro download_gmst
```

---
### II: Cleaning

We need to clean the datasets

---
### III: Feature Preparation

These steps will include: 

---
### IV: TRAINING

```bash
python st_evt/_src/modules/models/ts_stationary/aemet.py train-model-station --load-path "data/ml_ready"
```

---
### III: Visualization

We have a LOT of figures to do analysis.
So, I have split it up into specific instances.

#### EDA

The first is EDA. 


**Global Mean Surface Temperature Anomaly**


**STATIONS**

```bash
python st_evt/_src/modules/eda/aemet_station.py viz-t2max-station --load-path "data/clean" --save-path="/home/juanjohn/pool_data/dynev4eo/figures" --station-id='3129A'
```

**Block Maxima**

```bash
python st_evt/_src/modules/viz/bm/aemet.py viz-t2max-bm-year --load-path "data/ml_ready" --save-path="/home/juanjohn/pool_data/dynev4eo/figures" --station-id='3129A'
```

#### Features

The second is the

```bash
dvc repro
dvc exp run
```

**Plot Station EDA**

```bash
dvc exp run -s viz_t2max \
  --set-param eda.station_id='8414A' \
  --set-param filepaths.viz_eda_path="/home/juanjohn/pool_data/dynev4eo/figures" --force
```