# DARPA Mineral Competition - Point Features

This repo provides the code for `isi-umn` team in the [DARPA Mineral Competition](https://criticalminerals.darpa.mil/Leaderboard) 

Trained DNN Models can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1p86MHh_L3xilZNMw7jclt6Qpx_FCo4-n?usp=sharing)

## Overview: 
Point feature detection is performed with an assembly of three models: 1) color based vision model 2) shape based deep neural network model (DNN) and 3) shape based template matching model. The color-based model aims to detect symbols with colors, and the other two models aim to detect Black/White symbols. The difference between DNN and template matching models is the amount of training data. When there are multiple training maps available for one symbol, we train a DNN model. Otherwise we use template matching which does not require any training data. 

* **Color-based vision model**: Find the symbol locations by searching for similar RGB values same as template. (Does **not** require training)
* **DNN model**: A neural network classifier that takes in a small image patch and decides if it is the target symbol or not. (**Requries** training)
* **Template matching model**: A template matching model that looks for legend patterns on the map. (Does **not** require training)



## Code Structure

    .
    ├── Presentation1.pdf                # Slides to describe the method
    ├── apply_mask.py                    # Apply the map content mask on the predictions to generate final output
    ├── const_test.py                    # Some constant variables for test data 
    ├── const_val.py                     # Some constant variables for validation data
    ├── datasets.py                      # For DNN model: defines the dataset loader
    ├── models.py                        # For DNN model: constructs the model
    ├── run.sh                           # For DNN model: some sample scripts to train and test the DNN model
    ├── test_color.py                    # Testing code for color-based matching
    ├── test_dnn.sh                      # Testing code for DNN model
    ├── test_template.py                 # Testing code for template matching
    ├── train_points.py                  # Training code for DNN model
    └── README.md
    
    
##  
