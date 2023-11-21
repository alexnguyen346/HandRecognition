# This is a sample Python script.
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import json
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def train():
    # Dictionary of index and respective classes
    lookup = dict()
    reverselookup = dict()
    for count, label in enumerate(os.listdir('./leapGestRecog/leapGestRecog/00/')):
        if not label.startswith('.'):
            lookup[label] = count
            reverselookup[count] = label

    # Save dictionary of index and class
    with open("reverselookup.json", "w") as file:
        json.dump(reverselookup, file)

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation((-0.2, 0.3))])

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(200, 200, input_shape=[200, 200, 1]),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    images = []
    labels = []
    # Iterate through 10 sample folders
    for i in range(10):
        print(i)
        # Iterate through the different gestures of each sample
        for j in os.listdir(os.path.join('./leapGestRecog/leapGestRecog', '0' + str(i))):
            if not j.startswith('.'):
                # Iterate through the photos
                for k in os.listdir(os.path.join('./leapGestRecog/leapGestRecog', '0'+str(i), j)):
                    img_path = os.path.join('./leapGestRecog/leapGestRecog', '0'+str(i), j, k)
                    # Resize and normalize
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (200, 200))
                    img_normalized = img / 255.0

                    # Append to list of images and respective labels
                    images.append(img_normalized)
                    labels.append(lookup[j])

                    # Data augmentation

                    '''for _ in range(2):
                        img_expanded = np.expand_dims(img, axis=0)

                        augmented_image = data_augmentation(img_expanded)
                        augmented_image = augmented_image[0].numpy()

                        augmented_image = cv2.resize(augmented_image, (200, 200))
                        img_normalized = augmented_image / 255.0

                        # Append the preprocessed augmented image to the list
                        images.append(img_normalized)

                        # Assign label using lookup dictionary
                        labels.append(lookup[j])'''

    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    print("Finished loading data")

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', input_shape=(200, 200, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=len(lookup), activation='softmax')
    ])

    # Train on dataset
    model.compile(loss="sparse_categorical_crossentropy", optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # Save model
    model.save("trainedmodel.h5")


def predict():
    # Load the dictionary
    with open("reverselookup.json", "r") as file:
        reverselookup = json.load(file)

    # Load the trained model
    loaded_model = tf.keras.models.load_model("trainedmodel.h5")
    loaded_model.summary()
    images = []
    img_path = os.path.join('./testImages/test04.jpg')
    img = cv2.imread(img_path)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Binary image with some adjustments to take account of contrast intensities
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding
    normalized_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             15, 4)
    # Histogram equalization
    normalized_image = cv2.equalizeHist(normalized_image)

    # Contrast stretching
    min_intensity = np.min(normalized_image)
    max_intensity = np.max(normalized_image)
    normalized_image = (normalized_image - min_intensity) / (max_intensity - min_intensity) * 255.0

    # Resize and normalize
    resized_image = cv2.resize(normalized_image, (200, 200))
    img = resized_image / 255.0
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    images.append(img)
    images = np.array(images, dtype='float32')

    # Prediction
    prediction = loaded_model.predict(images)
    predicted_label_index = np.argmax(prediction)
    predicted_label = reverselookup[str(predicted_label_index)]
    print(predicted_label)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
