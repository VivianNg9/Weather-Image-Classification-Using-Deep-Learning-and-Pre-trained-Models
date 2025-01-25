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
  <summary>Click to view: Training Set Label Distribution:</summary>

| Image label |         |
| --------    | ------- |
| sunrise     | 255     |
| cloudy      | 208     |
| shine       | 179     |
| rain        | 142     |

Name: count, dtype: int64
</details>

<details>
  <summary>Click to view: Validation Set Label Distribution:</summary>

| Image label |         |
| --------    | ------- |
| sunrise     | 52      |
| cloudy      | 47      |
| shine       | 39      |
| rain        | 31      |

Name: count, dtype: int64
</details>

<details>
  <summary>Click to view: Test Set Label Distribution:</summary>

| Image label |         |
| --------    | ------- |
| sunrise     | 49      |
| cloudy      | 45      |
| shine       | 40      |
| rain        | 35      |

Name: count, dtype: int64
</details>

Visulizaing Label Distribution 
![Label Distribution](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Label%20Distribution.png)

- The `sunrise` label is the most frequent across all sets, which could lead to the model favoring predictions of this class. 
- The training, validation, and test sets maintain similar proportions of each label, ensuring that the model is evaluated on representative sample. This balanced split enhances the reliability of validation and test results, ensuring that the model generalizes well. 
- The validation and test sets contain a balanced representation of each label, ensuring a fair evaluation of model performance on unseen data.
- The model's ability to generalize across all classes, especially the minority ones, will be a critical aspect to monoritor. The high frequency `sunrise` images may cause the model to be overly confident in predicting this label, potentially skewing performance.

