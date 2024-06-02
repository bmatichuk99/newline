import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Streamlit interface
st.title('MNIST Digit Recognizer')

# Button to train the model
if st.button('Train Model'):
    with st.spinner('Training the model...'):
        model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    st.success('Model trained!')

# Canvas for drawing digits
st.write('Draw a digit below:')
canvas_result = st_canvas(stroke_width=15, stroke_color='#FFFFFF', background_color='#000000', height=280, width=280, drawing_mode='freedraw')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ImageOps.invert(Image.fromarray(img))
    img = np.array(img) / 255.0

    # Display the drawn image
    st.write('Your drawing:')
    st.image(img, width=140)

    # Button to test the model with the drawn digit
    if st.button('Test Model'):
        prediction = model.predict(img.reshape(1, 28, 28))
        st.write(f'Prediction: {np.argmax(prediction)}')

# Show the accuracy of the model
if st.button('Show Accuracy'):
    loss, accuracy = model.evaluate(x_test, y_test)
    st.write(f'Accuracy: {accuracy * 100:.2f}%')
