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

**Simple Model Summary**
- The model consists of two layers:
  - **Flatten Layer**: Converts the input image (230x230x3) into a 1D vector of shape (158,700)
  - **Dense Layer**: A fully connected layer with 4 neurons (one for each class), using the *softmax* activation function. This output layer provides probabilities for each class. 
- The model has a total of 634,804 trainable parameters. These parameters will be updated during training to optimize the model's performance on the classification task.

  **Training Loss**
- The training loss drop dramatically during the first epoch, from a very high value (>20) to around 5. This indicates that the model is quickly learning from the data in the initial stages.
- After the first epoch, the loss continues to decrease but with more fluctiations, suggesting that the model is still learning but the rate of improvement is slowing down.

**Validation Loss**
- The validation loss starts off much lower than the training loss but fluctutes over the epochs.
- The instability in the validation loss, especially around epochs 2 and 4, suggests that the model may be starting to *overfit* to the training data, as the loss does not consistently decrease.

**Training Accuracy**
- The training accuracy improves significantly from ~45% in the first epoch to around 75% by the fifth epoch.

**Validation Accuracy**
- The validation accuracy also improves over time, starting around 50% and reaching 70-75% by the fifth epoch. 
- The validation accuracy follows the training accuracy closely, indicating that the model is generalizing relatively well. However, the small drop in the final validation accuaracy suggests slight overfitting. 

**The simple model (`Model 1`) achieved an accuracy of `72.19%` of the predictions made on the unseen test dataset are correct.**

#### 2.2. A More Complex Classifier (`Model 2`)

Build the Complex Model 

**Number of Hidden Layers (num_hidden)**<p>
 - 1 to 3 hidden layers <p>
 - *Justification*: 1 to 3 layers are used to let the model detect patterns without being too complex. More layers can learn complicated patterns, but too many could memorize the data instead of learning from it (overfitting).<p>

**Sizes of Hidden Layers (hidden_size)**<p>
- From 32 to 512 neurons, in steps of 32 <p>
- *Justification*: The layers have between 32 to 512 neurons, adjustable by 32 each time. This range supports to find the spot between a model that's too simple or too complex.<p>

**Dropout Rate (dropout)**<p>
- 0.0 to 0.9 <p>
- *Justification*: Set between 0% to 90%, dropout randomly turns off some neurons during training. This helps the model be robust and not too dependent on any one piece of data.<p>

**Learning Rate (lrate)**<p>
-  1e-4 to 1e-1 <p>
- *Justification*: It varies from 1e-4 to 1e-1. This range is broad enough to find a learning rate that's not too slow (taking forever to learn) or too fast (missing the best solution).<p>
