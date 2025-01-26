# __<center> Weather Image Classification Using Deep Learning and Pre-trained Models </center>__

## __<center>Overview</center>__
[This project](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/Weather%20Image%20Classification%20Using%20Deep%20Learning%20and%20Pre-trained%20Models%20.ipynb) focuses on building an end-to-end image classification application to classify weather conditions using the Multi-class Weather Dataset (MWD).
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
 
![Label Distribution](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Label%20Distribution.png)

- The `sunrise` label is the most frequent across all sets, which could lead to the model favoring predictions of this class. 
- The training, validation, and test sets maintain similar proportions of each label, ensuring that the model is evaluated on representative sample. This balanced split enhances the reliability of validation and test results, ensuring that the model generalizes well. 
- The validation and test sets contain a balanced representation of each label, ensuring a fair evaluation of model performance on unseen data.
- The model's ability to generalize across all classes, especially the minority ones, will be a critical aspect to monoritor. The high frequency `sunrise` images may cause the model to be overly confident in predicting this label, potentially skewing performance.

#### 1.2. Preprocessing and Preparation 

- Resize images to a fixed shape and normalize to [0,1] for uniformity and efficient training.
- Convert image paths and labels from CSV files into numeric format for classification.

**Batch of Images**

![Batch of Images](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Batch%20of%20Images.png)

### 2. Simple Classifier
#### 2.1. First Classifier `Model 1`
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

**The simple model `Model 1` achieved an accuracy of `72.19%` of the predictions made on the unseen test dataset are correct.**

**Accuracy Test for `Model 1`**
| Class | Accuracy |
| -------- | ------- |
| Cloudy | 0.94 |
| Rain | 0.35 |
| Sunrise | 0.92 |
| Shine | 0.49 |

#### 2.2. A More Complex Classifier `Model 2`

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
</details>

<details>
  <summary>Click to view: Hyperparameter tuning with Keras Tuner:</summary>
  
```python
# Initialize the BayesianOptimization tuner
tuner = kt.BayesianOptimization(
    build_complex_model,
    objective='val_accuracy',
    max_trials=10,
    num_initial_points=2,
    overwrite=True
)


# Conduct the hyperparameter search using the training and validation datasets
tuner.search(
    train_dataset, 
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

# Retrieve the best hyperparameters 
best_hp_complex_model  = tuner.get_best_hyperparameters(num_trials=1)[0]
# Rebuild the best model
best_model_complex = build_complex_model(best_hp_complex_model)

# Train the best model
history_complex_model = best_model_complex.fit(
    train_dataset, validation_data=validation_dataset, epochs=10,
    callbacks=[keras.callbacks.EarlyStopping(patience=2)]
)

# Utilize the existing training_plot function for visualization
training_plot(['loss', 'accuracy'], history_complex_model);
```
</details>

![Complex Model](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Complex%20Model%20.png)

![Complex Model2](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/Complex%20Model2.png)

- The high number of parameters (`25,392,804`) shows that the model is complex, likely able to capture intricate patterns in the data. However, such complexity may require regularization (dropout) to avoid overfitting.
- The dense layer with `160 neurons` strikes a balance between capacity and overfitting. Keras Tuner identified this size as optimal during the hyperparameter search.

**Training Loss** 
- The training loss decreases significantly from approximately `12 to 1` over the first few epoches indicating that the model is learning quickly and effectively.
- After epoch 3, the training loss plateaus around `0.5`, showing that the model reaches a stable state after a few epochs.

**Validation Loss** 
- The validation loss starts around 2 and fluctuates slightly over the epochs but remains relatively stable after epoch 4.
- The small gap between training and validationloss => the model is not significantly overfitting, which is a good sign. 

**Training Accuracy**
- The training accuracy improves rapidly, reaching over 80% after a few epochs => The model learns the training data effectively.

**Validation Accuracy**
- The validation accuracy is initially lower than the training accuracy but increases steadily, peaking around `81.43%` after a few epochs. However, it fluctuates slightly, indicating that there could be small amounts of overfitting, but nothing severe.

**The complex model (`Model 2`) achieved an accuracy of 75.15% on the test dataset.**

**Accuracy Test for (`Model 2`)**
| Class | Accuracy |
| -------- | ------- |
| Cloudy | 0.65 |
| Rain | 0.94 |
| Sunrise | 0.96 |
| Shine | 0.43 |

- The model performs well on `sunrise` and `rain` classes but struggles with `cloudy` and `shine`. Improving the model’s ability to distinguish between these classes through augmentation, better feature extraction, or rebalancing the dataset could significantly enhance its performance.

  
#### 2.3. Error Analysis
![Error Analysis](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/error%20analysis.png)

- `Model 2` has a better accuracy on the test data, with an accuracy `75.15%`, compared to Model's test accuracy of `72.19%`. The more complex architecture allowed `Model 2` to better capture patterns and generalize to unseen data, resulturing in superior test performance.
- `Model 2` demonstrates a lower degree of overfirring, as evidenced by the smaller gap between training and validation accuracies (`5%` compared to `8% for Model 1`). This indicates that `Model 2` is more capable of generalizing to new data, and the use of dropour likely hleped mitigate overfitting by promoting better regularization.

### 3. A more complex classifier 
#### 3.1 Using ConvNets 

<details>
  <summary>Click to view: Build the model using ConvNets:</summary>
 
``` python
from tensorflow.keras import layers

# Build the model using ConvNets
img_size = (230, 230)  # Image size
num_classes = len(CLASS_NAMES)
def build_ConvNets (hp):
    model_ConvNets = keras.Sequential()
     # Initial Convolutional Block with Tunable Parameters
    model_ConvNets.add(layers.Conv2D(filters=hp.Int('initial_filters', 32, 512, step=32),
                            kernel_size=hp.Choice('initial_kernel_size', [3, 5]),
                            activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                            padding='same'))
    model_ConvNets.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Additional Convolutional Blocks
    # Add 1 to 3 additional Conv+Pooling blocks with tunable hyperparameters
    for i in range(hp.Int('num_conv_blocks', 1, 3)):
        model_ConvNets.add(layers.Conv2D(filters=hp.Int(f'filters_{i}', 32, 512, step=32),
                                kernel_size=hp.Choice(f'kernel_size_{i}', [3, 5]),
                                activation='relu', padding='same'))
        model_ConvNets.add(layers.MaxPooling2D(pool_size=hp.Choice(f'pool_size_{i}', [2, 3])))
    
    # Flattening layer
    model_ConvNets.add(layers.Flatten())
    
    # Dense Layers 
    # Add 1 to 3 additional Conv+Pooling blocks with tunable hyperparameters
    for i in range(hp.Int('num_dense_blocks', 1, 3)):
        model_ConvNets.add(layers.Dense(units=hp.Int(f'dense_units_{i}', min_value=32, max_value=512, step=32),
                               activation='relu'))
        model_ConvNets.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.9, step=0.1)))
    
    # Output Layer
    model_ConvNets.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))

    # Compile the model
    model_ConvNets.compile(optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='sparse_categorical_crossentropy', # for integers
                  metrics=['accuracy'])
    
    print(model_ConvNets.summary())
    return model_ConvNets
```
</details>

![ConvNets](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/ConvNets.png)

![model3](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/model%203.png)

**The ConvNets model achieved an accuracy of `88.76%` on the test dataset.**

**Accuracy Test for `ConvNets Model`**
| Class | Accuracy |
| -------- | ------- |
| Cloudy | 0.96 |
| Rain | 0.88 |
| Sunrise | 0.96 |
| Shine | 0.69 |

**Strong Overall Performance**: <p>
With a test accuracy of 88.76%, this ConvNets model performs exceptionally well, particularly for classes like `cloudy (96%)` and `sunrise (96%)`. The use of convolutional layers clearly helps capture important spatial features from the images.

**Regularization with Dropout**:<p>
 The dropout layer helps prevent overfitting, and the closeness of the training and validation accuracy indicates that the model generalizes well to unseen data.

**Class-Specific Challenges**:<p>
`Shine (69%)` could benefit from further improvement. Techniques like data augmentation (e.g., brightness adjustments or rotations) or class weighting could help the model learn more about this class.
`Rain (88%)` might benefit from enhanced feature extraction, possibly by adding more convolutional layers or using different kernel sizes to capture finer details.

#### 3.2 Using pre-trained models 
<details>
  <summary>Click to view: Build the model using MobileNet:</summary>
 
```python
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT_1 = 224
IMG_WIDTH_1 = 224
IMG_CHANNELS =3
# Build model using MobileNet
def build_MobileNet():
    # Load MobileNet with pre-trained ImageNet weights, excluding the top layer
    base_model = tf.keras.applications.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_HEIGHT_1, IMG_WIDTH_1, IMG_CHANNELS),
        pooling="avg"  # Use global average pooling
    )
    
    # Freeze the pre-trained model weights during training
    base_model.trainable = False

    # Create a new sequential model and add the pre-trained base model
    model_MobileNet = tf.keras.Sequential([base_model])
    model_MobileNet.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model_MobileNet.add(tf.keras.layers.Dropout(rate=0.5))

    # Add the output classification layer with a softmax activation
    model_MobileNet.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax'))

    # Compile the model
    model_MobileNet.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',  # for integer 
        metrics=['accuracy']
    )

    return model_MobileNet
model_MobileNet = build_MobileNet()
model_MobileNet.summary()
```
</details>

![MobileNet](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/MobileNet1.png)

![MobileNet1](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/MobileNet.png)

**Loss Plot**
- Both training and validation loss decrease consistently over the epochs.
- Validation loss stabilizes at around 0.3, indicating that the model generalizes well without overfitting. <p>

=> The steady improvement in both training and validation performance shows that the pre-trained MobileNet model, combined with the additional layers, is effective in learning the new dataset.

**Accuracy Plot**
- Training accuracy steadily improves and approaches 90%.
- Validation accuracy also follows a similar trend, stabilizing near 0.90. <p>

=> The close alignment between training and validation performance further indicates that overfitting is well-controlled because of using dropout and the learning rate schedule.

**The MobileNet model achieved an accuracy of `93.49%` on the test dataset.**

**Accuracy Test for `MobileNet Model`**
| Class | Accuracy |
| -------- | ------- |
| Cloudy | 0.90 |
| Rain | 1.00 |
| Sunrise | 0.98 |
| Shine | 0.86 |

`Rain` and `sunrise` are the most easily distinguishable classes for the model, likely due to strong visual cues in the data. `Cloudy` and `shine` could benefit from further refinement, possibly by augmenting the dataset or tweaking the learning rate schedule for better differentiation.

#### 3.3 Comparative Evaluation 

![MobileNet Model](https://github.com/VivianNg9/Weather-Image-Classification-Using-Deep-Learning-and-Pre-trained-Models/blob/main/image/MobileNet%20Model.png)

Based on the accuracy reports for different weather categories from MobileNet Model (task 3.2), it appears that the system performed best on `rain` with perfect accuracy **(1.0)**. The category `sunrise`, `cloudy` also had high accuracy, **0.98** and **0.9** relatively. The most difficult weather to detect was `shine`, with the lowest accuracy of **0.86**, indicating that the system was less effective at correctly identifying this category compared to others.

