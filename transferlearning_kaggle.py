# -*- coding: utf-8 -*-
"""TransferLearning_Kaggle.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dubtoIDhhPqLJVB48XK5zyRLzb8NrAfl

**Objective:**
Run prediction (inference) using pre-trained models and dataset from Kaggle.

***Imported the Necessary libraries***
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import os

# List the contents of the current directory
current_dir_contents = os.listdir('/content')

# Print the contents to verify the directory structure
print(current_dir_contents)

import os

# List the contents of the training and testing directories
train_contents = os.listdir(train_dir)
test_contents = os.listdir(test_dir)

print("Training directory contents:", train_contents)
print("Testing directory contents:", test_contents)

# Remove the .ipynb_checkpoints directories
!rm -r /content/dataset/train_set/.ipynb_checkpoints
!rm -r /content/dataset/test_set/.ipynb_checkpoints

"""**Downloaded the cat vs dog data set from kaggle and also Setting up the testing and training dataset directories**"""

# Set up data directories
train_dir = '/content/dataset1/train_set'  # Directory containing training images
test_dir = '/content/dataset1/test_set'    # Directory containing test images

"""**Data Preprocessing**

The code uses data generators to preprocess and augment image data for training and testing deep learning models. It rescales pixel values to the [0, 1] range, applies random rotations, shifts, and horizontal flips, and resizes images to a uniform size of 150x150 pixels. These techniques enhances the model robustness, enabling it to learn from varied perspectives and positions, ultimately improving classification accuracy.
"""

# Data Preprocessing
#train_datagen = ImageDataGenerator(rescale=1.0/255)
#test_datagen = ImageDataGenerator(rescale=1.0/255)

# New Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,  # Data augmentation: random rotation
    width_shift_range=0.2,  # Data augmentation: horizontal shift
    height_shift_range=0.2,  # Data augmentation: vertical shift
    horizontal_flip=True  # Data augmentation: horizontal flip
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

"""**Loading a pre trained model from kaggle**"""

# Loading a pre-trained model
model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False)
model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_generator, epochs=5, validation_data=test_generator)

"""**Evaluating the model**"""

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2%}")

"""As we can see, we are getting an accuracy of 76% after using the pre trained model from kaggle on the cat vs dog dataset downloaded from kaggle.

**Generating the confusion matrix**

The model exhibited strong performance by correctly identifying 23 cats and 35 dogs, demonstrating its ability to distinguish between the two classes. However, it also had shortcomings, misclassifying 27 cats as dogs and 19 dogs as cats. The confusion matrix, a vital evaluation tool, allows us to pinpoint areas of both success and improvement and provides key metrics for assessing binary classification model performance.
"""

# Generate confusion matrix
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype(int)
confusion_mtx = confusion_matrix(y_true, y_pred)
print(confusion_mtx)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate confusion matrix
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype(int)
confusion_mtx = confusion_matrix(y_true, y_pred)

# Define class labels
class_names = ['cat', 'dog']

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Sample Predictions and Accuracy
sample_images, sample_labels = next(test_generator)
sample_predictions = model.predict(sample_images)

for i in range(5):
    plt.figure()
    plt.imshow(sample_images[i])
    plt.title(f"True: {sample_labels[i]}, Predicted: {sample_predictions[i][0]:.2f}")
    plt.axis('off')

plt.show()

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_mtx)

# Display classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['cat', 'dog']))

"""**Observation**: Throughout this process, I've gained practical experience in image classification using the pre trained mobileNetV2 model from kaggle. Data preprocessing played a crucial role in setting the stage for model training, encompassing tasks like resizing, normalization, and data augmentation. I encountered and addressed challenges related to dataset organization, hidden files, and directory structure, which underscored the importance of maintaining a structured dataset. By experimenting with different pre-trained models and fine-tuning hyperparameters, I learned that model selection and training settings significantly impact accuracy. Monitoring training progress, analyzing confusion matrices, and assessing the model's ability to distinguish between classes provided valuable insights into performance.

"""