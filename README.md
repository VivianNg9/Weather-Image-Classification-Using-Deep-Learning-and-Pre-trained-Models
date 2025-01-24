# __<center> Weather Image Classification Using Deep Learning and Pre-trained Models </center>__

## __<center>Overview</center>__
This project focuses on building an end-to-end image classification application to classify weather conditions using the Multi-class Weather Dataset (MWD).
The application utilizes machine learning models, including simple classifiers, advanced convolutional neural networks (ConvNets), and pre-trained models like MobileNet. 
The objective is to demonstrate expertise in data preparation, model development, evaluation, and deployment using Python and TensorFlow.

## __<center>Dataset</center>__
The [`Multi-class Weather Dataset (MWD)`](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/dataset2.zip) consists of 1,125 labeled images representing various weather conditions:
- Classes: Cloudy, Shine, Rain, and Snow.
- Dataset distribution:
**Training:** 70% of the data used for training the model.
**Validation:** 15% of the data used to tune hyperparmeters and avoid overfitting.
**Test:** 15% of the data used for evaluating the model.
- The `random_state=42` ensures reproducibility of the splits.

## __<center>Project Environment</center>__
**Programming Language**: Python
**Frameworks and Libraries**: TensorFlow, Keras, Numpy, Pandas, Matplotlib. 

## __<center>Project Workflow</center>__
### 1. Data Partition, Preprocessing and Preparation 
1.1. Data Partition 
<details>
  <summary>Training Set Label Distribution:</summary>
 Image label
sunrise    255
cloudy     208
shine      179
rain       142
Name: count, dtype: int64
</details>

Validation Set Label Distribution:
 Image label
sunrise    52
cloudy     47
shine      39
rain       31
Name: count, dtype: int64

Test Set Label Distribution:
 Image label
sunrise    49
cloudy     45
rain       40
shine      35
Name: count, dtype: int64
