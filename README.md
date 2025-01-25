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
#### 1.1. Data Partition 
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

Visualize Label Distribution 
![Label Distribution](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Label%20Distribution.png)

- The `sunrise` label is the most frequent across all sets, which could lead to the model favoring predictions of this class. 
- The training, validation, and test sets maintain similar proportions of each label, ensuring that the model is evaluated on representative sample. This balanced split enhances the reliability of validation and test results, ensuring that the model generalizes well. 
- The validation and test sets contain a balanced representation of each label, ensuring a fair evaluation of model performance on unseen data.
- The model's ability to generalize across all classes, especially the minority ones, will be a critical aspect to monoritor. The high frequency `sunrise` images may cause the model to be overly confident in predicting this label, potentially skewing performance.

#### 1.2. Preprocessing and Preparation 
- By resizing the images to a fixed values to the [0,1] range, the data becomes uniform.
- Resizing all images to the same shape and normalizing their pixel values ensures consistency, which is important for efficient model training. A deep learning model typically performs better when input images are scaled to a fixed size and normalized.
- The CSV file contains paths to the images and their respective labels. This function converts those paths into images and ensurees the labels are transformed into numeric format, which is crucial for classification tasks. 

Visualize a Batch of Images 

![Batch of Images](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Batch%20of%20Images.png)

### 2. Simple Classifier
#### 2.1. First Classifier (`Model 1`)
<details>
  <summary>Click to view: Build a Simple Model :</summary>
  
```python
def build_simple_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CLASS_NAMES, lrate=0.001):
    model_simple = keras.Sequential([
        keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)), # Flatten the input image 
        keras.layers.Dense(len(CLASS_NAMES), activation='softmax') # Output layer with softmax activation 
    ])
    
    # Compile the model 
    model_simple.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model_simple.summary())
    return model_simple

simple_model = build_simple_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CLASS_NAMES)

#Train the model with early stopping callback to prevent overfitting
history_simple_model = simple_model.fit(train_dataset, validation_data=validation_dataset, epochs=10,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

# Utilize the existing training_plot function for visualization
training_plot(['loss', 'accuracy'], history_simple_model);
```
</details>
![Simple Model](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Simple%20Model.png)

![Simple Model1](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/simple%20model1.png)


