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


## I: Data Download

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

---
### II: Cleaning

We need to clean the datasets to make sure that they are harmonized and ready for analysis.

```bash
# Clean Everything
dvc repro pipelines/clean/dvc.yaml
# Clean Specifics
dvc repro pipelines/clean/dvc.yaml:clean_aemet_t2max
dvc repro pipelines/clean/dvc.yaml:clean_aemet_pr
dvc repro pipelines/clean/dvc.yaml:clean_gmst_david
```

---
### III: Feature Preparation

This section will be mainly extremes.

```bash
# Clean Everything
dvc repro pipelines/extremes/dvc.yaml
# Clean Specifics
dvc repro pipelines/extremes/dvc.yaml:aemet_t2m_bm_summer_extremes
dvc repro pipelines/extremes/dvc.yaml:viz_aemet_t2max_bm
dvc repro pipelines/extremes/dvc.yaml:aemet_pr_bm_fall_extremes
dvc repro pipelines/extremes/dvc.yaml:viz_aemet_pr_bm
```
