import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load the pre-trained MobileNetV2 model
trained_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                  include_top=False,
                                                  weights=None)

trained_model.trainable = True  # Un-Freeze all the pretrained layers of 'MobileNetV2' for training.

last_layer = trained_model.get_layer('out_relu')
last_layer_output = last_layer.output
x = GlobalAveragePooling2D()(last_layer_output)  # Add Global Average Pooling Layer
x = Dropout(0.8)(x)  # Add a Dropout layer
x = Dense(120)(x)
output = Dense(105, activation="softmax")(x)
model_mobilenet = Model(trained_model.input, output)

model_mobilenet.load_weights('model_mobilenet.h5')

model_mobilenet_facenet = Model(model_mobilenet.input, model_mobilenet.get_layer('dense_1').output)  # Removing the last classification layer


def calculate_embeddings(path, model, input_shape):
    embeddings = []  # It will be a list of lists; will store lists of type [embedding vector, label of image]
    for folder in os.listdir(path):
        text = folder
        folder_path = os.path.join(path, text)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img = tf.expand_dims(img, axis=0)
            img_predict = model.predict(img)
            emb_item = [img_predict, text]
            embeddings.append(emb_item)
    return embeddings

# Calculate embeddings of dataset
embeddings = calculate_embeddings("images/", model_mobilenet_facenet, (160, 160))

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()
    
    if not ret:
        print("Failed to capture image from camera")
        break

    start_time = time.time()  # Start time for FPS calculation

    # Detect the faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around detected faces and recognize them
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_LINEAR)
        face_img = tf.expand_dims(face_img, axis=0)
        img_predict_2 = model_mobilenet_facenet.predict(face_img)  # Create embeddings of the detected face

        max_similarity = 0.0
        recognized_label = "Unknown"

        # Check for similarity with embeddings of the images in the database
        for embedding in embeddings:
            val = cosine_similarity(embedding[0], img_predict_2)
            if val > max_similarity:
                max_similarity = val
                recognized_label = embedding[1]

        # Check if maximum similarity crosses the threshold or not
        if max_similarity < 0.8:
            recognized_label = "Unknown"

        # Display recognized label with similarity score
        text = f"{recognized_label} ({max_similarity[0][0]:.2f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
