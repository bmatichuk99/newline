import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import cv2

# Set up the drawing canvas
st.title("MNIST Digit Recognizer with Data Augmentation")
st.sidebar.header("Train the Model")

# Set canvas stroke width and size
stroke_width = 18
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Function to preprocess the canvas image for prediction
def preprocess_image(image):
    # Convert to grayscale and resize to 28x28
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to build and compile the model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the MNIST data and apply data augmentation
def load_and_augment_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)
    return datagen, x_train, y_train, x_test, y_test

# Train the model
if st.sidebar.button("Train Model"):
    st.sidebar.text("Training the model...")
    datagen, x_train, y_train, x_test, y_test = load_and_augment_data()
    model = build_model()
    model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test))
    model.save("mnist_model.keras")
    st.sidebar.success("Model trained and saved!")

# Load the model for prediction
if st.sidebar.button("Load Model"):
    model = models.load_model("mnist_model.keras")
    st.sidebar.success("Model loaded!")

# Make a prediction if a drawing is made
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    processed_image = preprocess_image(image)
    st.image(processed_image.reshape(28, 28), caption="Preprocessed Image", width=150)
    if 'model' in locals():
        prediction = model.predict(processed_image)
        st.write(f"Predicted Digit: {np.argmax(prediction)}")
    else:
        st.write("Please load a trained model first.")

# Layout settings to fit everything nicely on one page
st.sidebar.markdown("---")
st.sidebar.write("Adjust layout to fit everything nicely.")
