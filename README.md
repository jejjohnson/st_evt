# Spatio-Temporal Extreme Values

> In this repo we


## Pipelines

> We have ready-made data that's ready to go for exploration and modeling fitting.

**Download Everything**

```bash
dvc repro pipelines/download/dvc.yaml
```

**Clean Everything**

```bash
dvc repro pipelines/clean/dvc.yaml
```

**Feature Preparation (Extremes)**

```bash
dvc repro pipelines/extremes/dvc.yaml
```


## Installation

First, clone the repo

```bash
git clone https://github.com/jejjohnson/st_evt.git
cd st_evt
```


Install using `conda`.

```bash
conda env create -f environments/environment.yaml
conda activate stevt
```

We also have a gpu environment available in case you want to use GPUs. However, it's not necessary unless you're personally interesting. Everything runs just fine on CPU.