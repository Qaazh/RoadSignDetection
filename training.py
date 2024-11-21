# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For plotting accuracy and loss graphs
import cv2  # For image processing
import tensorflow as tf  # For building the neural network
from PIL import Image  # For handling image files
import os  # For navigating the file system
from sklearn.model_selection import train_test_split  # For splitting the dataset
from keras.utils import to_categorical  # For one-hot encoding of labels
from keras.models import Sequential  # For creating a sequential model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # For creating CNN layers

# Initialize lists to store data (images) and labels (traffic sign classes)
data = []
labels = []
classes = 43  # Total number of classes
cur_path = os.getcwd()  # Get the current directory path

# Load the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))  # Set path for each class
    images = os.listdir(path)  # List all image files in the class directory
    for a in images:
        try:
            # Open the image, resize it to 30x30 pixels, convert to a numpy array, and add to the list
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)  # Print the shape of the data and label arrays

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Create the CNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compile the model with the categorical crossentropy loss function and the Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with the training data, validate with the testing data, and set epochs
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# Save the trained model
model.save("traffic_classifier.h5")

# Plot accuracy and loss for training and validation
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the model accuracy on a separate test dataset
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
test_data = []
for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    test_data.append(np.array(image))
X_test = np.array(test_data)

# Predict the classes using the model
pred_probabilities = model.predict(X_test)  # Get probabilities for each class
pred_classes = np.argmax(pred_probabilities, axis=-1)  # Convert probabilities to class labels

# Print the accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred_classes))

# Save the model again (optional, as it's already saved above)
model.save('traffic_classifier.keras')
