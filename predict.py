import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import argparse

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Define function to predict the digit
def predict_digit(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image

    # Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    print(f"Predicted digit: {predicted_digit}")

# Parse image path from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a digit from an image.")
    parser.add_argument('--image', required=True, help="Path to the image to predict")
    args = parser.parse_args()

    # Make prediction
    predict_digit(args.image)
