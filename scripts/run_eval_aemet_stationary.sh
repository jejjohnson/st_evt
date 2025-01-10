posterior_stations=(
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

posterior_predictive_stations=(
    '1391'
    '2227'
    '2894'
    '3107A'
    '3257'
    '3403B'
    '3416'
    '3428'
    '5826U'
    '7262E'
    '9445E'
    '9735E'
    '9839O'
    '9988D'
    '0139I'
    '0365D'
    '1044'
    '2294'
    '2409E'
    '3075'
    '4203E'
    '4210'
    '5641A'
    '6349'
    '7106'
    '9282A'
    '9433U'
    '9434M'
)

# #######################
# # IID MODEL - MCMC
# #######################

# for i in "${posterior_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/stations' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# # REGION
# python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
#     --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/stationary_iid_mcmc_redfeten.zarr' \
#     --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_mcmc_redfeten/results/figures/spain/'

# #######################
# # IID MODEL - LAPLACIAN
# #######################

# for i in "${posterior_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_lap_redfeten/results/stationary_iid_lap_redfeten.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_lap_redfeten/results/figures/stations' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# # REGION
# python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
#     --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_lap_redfeten/results/stationary_iid_lap_redfeten.zarr' \
#     --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/stationary_iid_lap_redfeten/results/figures/spain/'


# #######################
# # GP MODEL - MCMC
# #######################


# # STATIONS - Posterior
# for i in "${posterior_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/az_stationary_gp_mcmc.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/figures/stations/' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# # STATIONS - Posterior
# for i in "${posterior_predictive_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-predictive-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/az_stationary_gp_mcmc.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/figures/stations/' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# REGION
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/az_stationary_gp_mcmc.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/figures/spain/' \

# # REGION - GP PARAMETERS
# python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-gp-params \
#     --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/az_stationary_gp_mcmc.zarr' \
#     --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_mcmc_redfeten/results/figures/spain/' \


# #######################
# # GP MODEL - LAPLACIAN
# #######################

# # STATIONS - Posterior
# for i in "${posterior_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/stations/' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# # STATIONS - Posterior
# for i in "${posterior_predictive_stations[@]}"; do
#     python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-predictive-station \
#         --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
#         --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/stations/' \
#         --dataset-url="/home/juanjohn/projects/st_evt/data/ml_ready/aemet/t2max_stations_bm_summer.zarr" \
#         --station-id="$i"
# done

# REGION
python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-region \
    --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
    --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/spain/'

# # REGION - GP PARAMETERS
# python st_evt/_src/modules/models/aemet/eval_stationary_model.py evaluate-model-posterior-gp-params \
#     --results-dataset='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/az_stationary_gp_lap_redfeten.zarr' \
#     --figures-path='/home/juanjohn/pool_data/dynev4eo/experiments/walkthrough/aemet/t2max/az_stationary_gp_lap_redfeten/results/figures/model_params/'