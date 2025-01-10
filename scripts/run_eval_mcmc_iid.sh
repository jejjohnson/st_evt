strings=(
    '3129A'
    '8414A'
    '9434'
    '1475X'
    '7178I'
    '3407Y'
    '9677'
    'C018J'
    'C426R'
    '2044B'
    '2401X'
    '2661B'
    '3094B'
)

# for i in "${strings[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/stations' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# STATIONS - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/spain/'

# REGION - Posterior Predictive
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/spain/'