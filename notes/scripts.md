


**Run Model-Train Script**

```bash
python st_evt/_src/modules/models/ts_nonstationary/aemet.py train-model-station \
    --load-path data/ml_ready \
    --save-path /home/juanjohn/pool_projects/scratch \
    --station-id "8414A"
```

**Run Model-Predict Script**

```bash
python st_evt/_src/modules/models/ts_nonstationary/aemet.py evaluate-model-station \
    --load-path /home/juanjohn/pool_projects/scratch \
    --save-path /home/juanjohn/pool_projects/scratch
```


**GEVD Stationary GP - Laplacian**

```bash
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten_clean/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten_trainpred/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --include-pred-noise \
    --red-feten
```

**GEVD Stationary GP - MCMC**

```bash
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines" \
    --num-map-warmup=50_000 \
    --num-mcmc-samples=1_000 \
    --num-chains=8 \
    --num-mcmc-warmup=1_000
```


**GEVD NonStationary GP - Laplace Approximation**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_clean/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainpred/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --include-pred-noise \
    --red-feten
```

**GEVD NonStationary GP - MCMC**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten/results" \
    --include-train-noise \
    --red-feten \
    --num-map-warmup=50_000 \
    --num-mcmc-samples=1_000 \
    --num-chains=12 \
    --num-mcmc-warmup=5_000
```


## Model Evaluation

### Stationary Models

#### IID MODELS

##### MCMC Inference

```bash
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/stations' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='8414A'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/spain/'
# REGION - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/spain/'
```

#### Gaussian Process Models

- CANDIDATE STATIONS - '3129A', '8414A', '9434', '1475X', '7178I'
- POSTERIOR - STATIONS - '3407Y', '9677', 'C018J', 'C426R', '2044B', '2401X', '2661B', '3094B'
- POSTERIOR PREDICTIONS - STATIONS - 
BAD STATIONS
'1391',
  '2227',
  '2894',
  '3107A',
  '3257',
  '3403B',
  '3416',
  '3428',
  '5826U',
  '7262E',
  '9445E',
  '9735E',
  '9839O',
  '9988D'
GOOD STATIONS
'0139I',
  '0365D',
  '1044',
  '2294',
  '2409E',
  '3075',
  '4203E',
  '4210',
  '5641A',
  '6349',
  '7106',
  '9282A',
  '9433U',
  '9434M'

###### MCMC Inference

```bash
# STATIONS - Posterior
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/stations/posterior_predictive/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='C018J'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-predictive-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/stations/predictions/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='0139I'
# REGIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/spain/'
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-gp-params \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/model_params/'
```

###### Laplace Approximation


```bash
# STATIONS - Posterior
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/stations/posterior/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='C018J'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-predictive-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/stations/posterior_predictive/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='0139I'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/spain/'
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-gp-params \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/model_params/'
```


---
### NonStationary Models

# POSTERIOR - STATIONS - '3407Y', '9677', 'C018J', 'C426R', '2044B', '2401X', '2661B', '3094B'

#### IID MODELS

#### MCMC Inference

```bash
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/nonstationary_iid_mcmc_redfeten/results/nonstationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/nonstationary_iid_mcmc_redfeten/results/figures/stations/posterior' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id="3129A"
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-predictive-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/nonstationary_iid_mcmc_redfeten/results/nonstationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/stations/predictions/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id="3129A"
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/nonstationary_iid_mcmc_redfeten/results/nonstationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/nonstationary_iid_mcmc_redfeten/results/figures/spain/'
```

#### Gaussian Process Models

- POSTERIOR - STATIONS - '3407Y', '9677', 'C018J', 'C426R', '2044B', '2401X', '2661B', '3094B'
- POSTERIOR PREDICTIONS - STATIONS - 
BAD STATIONS
'1391',
  '2227',
  '2894',
  '3107A',
  '3257',
  '3403B',
  '3416',
  '3428',
  '5826U',
  '7262E',
  '9445E',
  '9735E',
  '9839O',
  '9988D'
GOOD STATIONS
'0139I',
  '0365D',
  '1044',
  '2294',
  '2409E',
  '3075',
  '4203E',
  '4210',
  '5641A',
  '6349',
  '7106',
  '9282A',
  '9433U',
  '9434M'

###### MCMC Inference

```bash
# STATIONS - Posterior
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/stations/posterior_predictive/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='C018J'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-predictive-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/stations/predictions/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='0139I'
# REGIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/spain/'
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-gp-params \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/az_nonstationary_gp_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten_trainnoise/results/figures/model_params/'
```

###### Laplace Approximation


```bash
# STATIONS - Posterior
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/az_nonstationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/figures/stations/posterior_predictive/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='C018J'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-predictive-station \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/az_nonstationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/figures/stations/predictions/' \
    --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --station-id='0139I'
# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/az_nonstationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/figures/spain/'
python st_evt/_src/modules/models/aemet/eval_nonstationary_model.py evaluate-model-posterior-gp-params \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/az_nonstationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results/figures/model_params/'
```




---
### GEVD Stationary GP

This modue

**TRAIN LAPLACIAN MODEL**

```bash
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/gevd_stationary_gp_lap_redfeten/noise_exp/train/" \
    --include-train-noise \
    --red-feten \
    --num-iterations=10 \
    --num-posterior-samples=100
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/gevd_stationary_gp_lap_redfeten/noise_exp/train/" \
    --include-train-noise \
    --red-feten \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 
```

**TRAIN MCMC MODEL** (TODO)

```bash
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines" \
    --num-map-warmup=10 \
    --num-mcmc-samples=10 \
    --num-chains=4 \
    --num-mcmc-warmup=10
python st_evt/_src/modules/models/aemet/gevd_stationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/" \
    --num-map-warmup=50_000 \
    --num-mcmc-samples=1_000 \
    --num-chains=8 \
    --num-mcmc-warmup=2_000
```

---
### GEVD Non-Stationary GP

This modue

**TRAIN LAPLACIAN MODEL**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results" \
    --num-iterations=10 \
    --num-posterior-samples=10 \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/noise_exp/clean/" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/gevd_nonstationary_gp_lap_redfeten/noise_exp/train/" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
```

**TRAIN MCMC MODEL** (TODO)


**Test Case**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten/results" \
    --include-train-noise \
    --include-pred-noise \
    --red-feten \
    --num-map-warmup=10 \
    --num-mcmc-samples=10 \
    --num-chains=4 \
    --num-mcmc-warmup=10
```


**Actual Experiment**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-mcmc \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten/results" \
    --include-train-noise \
    --red-feten \
    --num-map-warmup=50_000 \
    --num-mcmc-samples=1_000 \
    --num-chains=12 \
    --num-mcmc-warmup=1_000
```


---
## Demo Experiment

### Experiment IIa - Laplacian Model


**TRAIN LAPLACIAN MODEL**

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_trainnoise/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --include-train-noise \
    --red-feten
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_train.py train-model-laplace \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --save-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_lap_redfeten_clean/results" \
    --num-iterations=50_000 \
    --num-posterior-samples=1_000 \
    --red-feten
```


**EVALUATE MODEL (STATION)**

##### Predictive Posterior Analysis

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_eval_station.py evaluate-model-station-posterior \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --results-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/gevd_nonstationary_gp_lap_redfeten/noise_exp/train/az_nonstationary_gp_lap_redfeten.zarr" \
    --figures-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/figures/nonstationary_gp_lap_redfeten/temp" \
    --station-id='1391'
```

##### Predictions

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_eval_station.py evaluate-model-station-predictions \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --results-path="/home/juanjohn/pool_data/dynev4eo/temp/results/scratch_pipelines/results/gevd_nonstationary_gp_lap_redfeten/noise_exp/train/az_nonstationary_gp_lap_redfeten.zarr" \
    --figures-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/figures/nonstationary_gp_lap_redfeten/temp" \
    --station-id='1391'
```

### Experiment IIb - MCMC Model

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_eval_station.py evaluate-model-station-posterior \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --results-path="/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_nonstationary_gp_mcmc_redfeten/results" \
    --figures-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/figures/nonstationary_gp_mcmc_redfeten/temp" \
    --include-train-noise \
    --red-feten \
    --num-map-warmup=50_000 \
    --num-mcmc-samples=1_000 \
    --num-chains=8 \
    --num-mcmc-warmup=1_000
```

##### Predictive Posterior Analysis

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_eval_station.py evaluate-model-station-posterior \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --results-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/results/az_nonstationary_gp_mcmc_redfeten.zarr" \
    --figures-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/figures/nonstationary_gp_mcmc_redfeten/temp" \
    --station-id='1391'
```

##### Predictions

```bash
python st_evt/_src/modules/models/aemet/gevd_nonstationary_gp/model_eval_station.py evaluate-model-station-predictions \
    --dataset-path="data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
    --results-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/results/az_nonstationary_gp_mcmc_redfeten.zarr" \
    --figures-path="/home/juanjohn/pool_data/dynev4eo/temp/scratch_pipelines/figures/nonstationary_gp_mcmc_redfeten/temp" \
    --station-id='1391'
```