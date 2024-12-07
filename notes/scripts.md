


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