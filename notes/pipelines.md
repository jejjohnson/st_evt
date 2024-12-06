---
title: Pipelines & Experiments
subject: Tutorials
short_title: Pipelines
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


We have a few easy-to-use tools for users to get started right away.


## Data

First, we need to download some data.
To download data, we have some already done scripts that enable users to download data right away.

```bash
# Download Everything
dvc repro pipelines/download/dvc.yaml
```

**AEMET**

These are the weather stations that are available for the entire region in Spain.

```bash
# Download Everything
dvc repro pipelines/download/dvc.yaml
# Download Specifics
dvc repro pipelines/download/dvc.yaml:download_aemet
dvc repro pipelines/download/dvc.yaml:download_gmst

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

**Simple Time Series**

```bash
# TRAIN
dvc exp run pipelines/models/timeseries/dvc.yaml:aemet_stations_mcmc_train
# EVALUATION
dvc exp run pipelines/models/timeseries/dvc.yaml:aemet_stations_mcmc_evaluate
```

**Simple Time Series**

```bash
# TRAIN
dvc exp run pipelines/models/timeseries/dvc.yaml:aemet_stations_mcmc_train --set-param eda.station_id='8414A'
# EVALUATION
dvc exp run pipelines/models/timeseries/dvc.yaml:aemet_stations_mcmc_evaluate
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
dvc exp run pipelines/viz/dvc.yaml:viz_station_t2max --set-param eda.station_id='3129A' --downstream --force
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