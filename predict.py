import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

def process_image(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  return (image / 255).numpy()

def predict(image_path, model, top_k):
    with Image.open(image_path) as img:
        img = np.asarray(img)
        processed_image = process_image(img)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    probabilities, classes = tf.nn.top_k(prediction, k=top_k)
    return probabilities.numpy()[0], classes.numpy()[0]

def format_output(probabilities, classes, class_names=None):
    print("\nPredicted Classes and Probabilities:")
    print("="*40)
    for i, (prob, class_) in enumerate(zip(probabilities, classes)):
        class_name = class_names[str(class_ )] if class_names else class_
        print(f"{i+1:>2}. Class: {class_name:<20} | Probability: {prob:.4f}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description='Predict the class of a flower image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the saved model.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to display.')
    parser.add_argument('--category_names', type=str, help='Path to the JSON file mapping class numbers to names.')
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    probabilities, classes = predict(args.image_path, model, args.top_k)
    class_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    format_output(probabilities, classes, class_names)

if __name__ == '__main__':
    main()
