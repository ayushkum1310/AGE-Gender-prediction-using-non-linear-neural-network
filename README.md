# Age and Gender Prediction using Multi-Output Model with VGG16

This repository contains a TensorFlow/Keras implementation for predicting age and gender simultaneously using a multi-output neural network model based on VGG16. The model is trained on the UTKFace dataset.

## Overview

- **Dataset:** UTKFace dataset consisting of images labeled with age and gender.
- **Model Architecture:** 
  - VGG16 base with pre-trained weights (excluding top layers).
  - Added fully connected layers for feature extraction.
  - Separate outputs for predicting age (regression) and gender (binary classification).

## Key Steps

1. **Data Preparation:**
   - Loaded dataset and split into training and testing sets.
   - Applied data augmentation using `ImageDataGenerator` for training set.

2. **Model Building:**
   - Created a custom generator to handle multi-output labels.
   - Wrapped the generator using `tf.data.Dataset.from_generator` for compatibility.
   - Defined model architecture using VGG16 and added dense layers for predictions.

3. **Model Compilation and Training:**
   - Compiled the model with appropriate loss functions and metrics for age and gender predictions.
   - Trained the model using `model.fit` on the wrapped datasets (`train_dataset` and `test_dataset`).

4. **Results Visualization:**
   - Plotted training and validation losses.
   - Visualized Mean Absolute Error (MAE) for age prediction.
   - Analyzed accuracy for gender prediction.

5. **Conclusion and Recommendations:**
   - Summarized model performance.
   - Suggested improvements such as increasing training data, trying different models, and applying regularization.


## Files

- **Model File:** `AGE_Gender_10_epoc.h5` - Saved model file for deployment or further training.

## Regards </br>Ayush kumar</br>Data scientist
