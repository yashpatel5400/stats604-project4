![](https://onemontgomerygreen.org/wordpress/wp-content/uploads/2019/03/NEW_NOAA-Open-House-Banner_030419.jpg)


Weather Prediction (STATS 604)
========================================
![tests](https://github.com/prob-ml/bliss/workflows/tests/badge.svg)

# Introduction

This weather prediction repo is a collection of methods for predicting min, mean, and max temperatures. This repo provides
  - __Accurate estimation__ validated through CV over historical data.
  - __Reproducibility__ through an accompanying Dockerfile.

# Installation

1. To use and install `predictor`, we recommend using Docker. Build the image with: `docker build -f Dockerfile .`

2. Drop into the Docker shell with: `docker run -it --rm [IMAGE_NAME]`

3. To recreate the data before making a prediction, run
```
make clean
make rawdata
make
```
Note that this will already have been done in the initial Docker image construction.

4. Run the prediction with:
```
make predictions
```

# Code Structure
Code is organized into the following directories:
- `data/`: code for processing the raw data (currently support NOAA and Wunderground)
- `predictor/`: main source for the repo, which notably includes:
    - `predictor/eval.py`: evaluation pipeline used for model experimentation and validation
    - `predictor/models/`: collection of models tested for predictive accuracy
- `raw_data/`: code for downloading raw data (currently support NOAA and Wunderground)

# Authors
Vinod Raman, Unique Subedi, Seamus Somerstep, and Yash Patel
