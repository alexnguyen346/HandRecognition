# This is a sample Python script.
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
import json
import mediapipe as mp
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def train():
    # Dictionary of index and respective classes
    lookup = dict()
    reverselookup = dict()
    for count, label in enumerate(os.listdir('./archive/train/train/')):
        if not label.startswith('.'):
            lookup[label] = count
            reverselookup[count] = label

    # Save dictionary of index and class
    with open("reverselookup.json", "w") as file:
        json.dump(reverselookup, file)

    print(lookup)

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
    for j in os.listdir(os.path.join('./archive/train/train/')):
        print(j)
        if not j.startswith('.'):
            for k in os.listdir(os.path.join('./archive/train/train', j)):
                img_path = os.path.join('./archive/train/train', j, k)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (200, 200))
                grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_normalized = grayscale_image / 255.0
                images.append(img_normalized)
                labels.append(lookup[j])
    '''for i in range(10):
        print(i)
        # Iterate through the different gestures of each sample
        for j in os.listdir(os.path.join('./leapGestRecog/leapGestRecog', '0' + str(i))):
            if not j.startswith('.'):
                # Iterate through the photos
                for k in os.listdir(os.path.join('./leapGestRecog/leapGestRecog', '0'+str(i), j)):
                    img_path = os.path.join('./leapGestRecog/leapGestRecog', '0'+str(i), j, k)
                    # Resize and normalize
                    img = cv2.imread(img_path)

                    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Adaptive thresholding
                    normalized_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                             cv2.THRESH_BINARY,
                                                             15, 4)
                    # Histogram equalization
                    normalized_image = cv2.equalizeHist(normalized_image)

                    # Contrast stretching
                    min_intensity = np.min(normalized_image)
                    max_intensity = np.max(normalized_image)
                    normalized_image = (normalized_image - min_intensity) / (max_intensity - min_intensity) * 255.0

                    # Resize and normalize
                    resized_image = cv2.resize(normalized_image, (200, 200))
                    img_normalized = resized_image / 255.0
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (200, 200))
                    img_normalized = img / 255.0

                    # Append to list of images and respective labels
                    images.append(img_normalized)
                    labels.append(lookup[j])

                    # Data augmentation

                    for _ in range(2):
                        img_expanded = np.expand_dims(img, axis=0)

                        augmented_image = data_augmentation(img_expanded)
                        augmented_image = augmented_image[0].numpy()

                        augmented_image = cv2.resize(augmented_image, (200, 200))
                        img_normalized = augmented_image / 255.0

                        # Append the preprocessed augmented image to the list
                        images.append(img_normalized)

                        # Assign label using lookup dictionary
                        labels.append(lookup[j])'''

    print(1)
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
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

    # Save model
    model.save("trainedmodel.h5")


cap = cv2.VideoCapture(0)


def predict(img, loaded_model):

    # Resize and normalize
    resized_image = cv2.resize(img, (200, 200))
    img = resized_image / 255.0
    img = np.reshape(img, (1, 200, 200, 1))

    # Prediction
    prediction = loaded_model.predict(img)
    predicted_label_index = np.argmax(prediction)
    predicted_label = reverselookup[str(predicted_label_index)]
    return predicted_label, img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
    # Load the dictionary
    with open("reverselookup.json", "r") as file:
        reverselookup = json.load(file)

    # Load the trained model
    loaded_model = tf.keras.models.load_model("trainedmodel.h5")
    loaded_model.summary()

    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:
        thresh = False
        predicted_label = "None"
        success, img = cap.read()
        #img = cv2.flip(img, 1)
        img2 = img.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks_xy = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in handLms.landmark]

                # Compute bounding box
                x, y, width, height = cv2.boundingRect(np.array(landmarks_xy))
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # if id ==0:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                if width > 0 and height > 0:
                    # Crop the hand region
                    hand_crop = img2[y:y + height + 30, x - 30:x + width + 30]

                    # Check if the cropped image is not empty
                    if hand_crop.size > 0:
                        # Display the cropped hand region
                        hsvim = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
                        lower = np.array([0, 10, 30], dtype="uint8")
                        upper = np.array([255, 100, 150], dtype="uint8")
                        skinMask = cv2.inRange(hsvim, lower, upper)

                        # get threshold image
                        ret, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)

                        #thresh = cv2.resize(thresh, (200, 200))
                        #thresh = thresh / 255.0

                        cv2.imshow("thresh", thresh)

                        # Create a binary mask of the hand based on skin color

                    else:
                        thresh = False
        if thresh is not False:
            predicted_label, bg_img = predict(thresh, loaded_model)
            print(predicted_label)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.putText(img, f'Gesture: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the BGR image to RGB
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        skinMask = cv2.inRange(hsvim, lower, upper)

        # blur the mask to help remove noise
        skinMask = cv2.blur(skinMask, (2, 2))

        # adaptive thresholding
        thresh = cv2.adaptiveThreshold(skinMask, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Display the gesture label on the webcam window
        #predicted_label, bg_img = predict(img, loaded_model)

        #cv2.putText(frame, f'Gesture: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', thresh)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

'''def predict2():
    # Load the dictionary
    with open("reverselookup.json", "r") as file:
        reverselookup = json.load(file)

    # Load the trained model
    loaded_model = tf.keras.models.load_model("trainedmodel.h5")
    loaded_model.summary()
    images = []
    img_path = os.path.join('archive/train/train/1/1.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 125))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    cv2.imshow("Skin YCrCb", skin_ycrcb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # draw the contours on the empty image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)
    cv2.imshow("contours", img)


    # Binary image with some adjustments to take account of contrast intensities
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding
    normalized_image = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             31, 10)
                                             
    # Histogram equalization
    normalized_image = cv2.equalizeHist(normalized_image)

    # Contrast stretching
    min_intensity = np.min(normalized_image)
    max_intensity = np.max(normalized_image)
    normalized_image = (normalized_image - min_intensity) / (max_intensity - min_intensity) * 255.0

    # Resize and normalize
    resized_image = cv2.resize(img, (200, 200))
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
    predict2()'''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
