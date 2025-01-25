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

Visualize Label Distribution 
![Label Distribution](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Label%20Distribution.png)

- The `sunrise` label is the most frequent across all sets, which could lead to the model favoring predictions of this class. 
- The training, validation, and test sets maintain similar proportions of each label, ensuring that the model is evaluated on representative sample. This balanced split enhances the reliability of validation and test results, ensuring that the model generalizes well. 
- The validation and test sets contain a balanced representation of each label, ensuring a fair evaluation of model performance on unseen data.
- The model's ability to generalize across all classes, especially the minority ones, will be a critical aspect to monoritor. The high frequency `sunrise` images may cause the model to be overly confident in predicting this label, potentially skewing performance.

#### 1.2. Preprocessing and Preparation 

- Resize images to a fixed shape and normalize to [0,1] for uniformity and efficient training.
- Convert image paths and labels from CSV files into numeric format for classification.

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

<details>
  <summary>Click to view: Build the Complex Model:</summary>
  
```python
def build_complex_model(hp):
    # Adapting the input shape and CLASS_NAMES based on the current task's dataset
    model_complex = keras.Sequential()
    model_complex.add(keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    # Tuning the number of hidden layers and sizes, with dropout
    for i in range(hp.Int('num_hidden', 1, 3)):
        # The number of neurons in each layer to vary between 32 and 512, increasing in step of 32
        model_complex.add(keras.layers.Dense(hp.Int(f'hidden_size_{i}', 32, 512, step=32), activation='relu')) # 'relu'introduces non-linearity and helps the model learn complex patterns.
        model_complex.add(keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.9))) # Tuning dropout rate 
    model_complex.add(keras.layers.Dense(len(CLASS_NAMES), activation='softmax')) # 'softmax' ensures that the output is a probability distribution across the classes.

    # Compile the model 
    model_complex.compile(
        optimizer=optimizers.Adam(learning_rate=hp.Float('lrate', 1e-4, 1e-1, sampling='log')),
        loss=losses.SparseCategoricalCrossentropy(), # 'SparseCategoricalCrossentropy': for multi-class classification with integer labels. 
        metrics=['accuracy'])
    
    print(model_complex.summary())
    return model_complex
```
</detail>

