from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load your trained model (replace with the actual path if different)
model = load_model('Path of your model')

def predict_age_and_gender(image_path):
    """
    Predicts age and gender from a single image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the predicted age (float) and gender (int: 0 for male, 1 for female).
    """
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(200, 200))  # Match the model's input size
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x / 255.0  # Rescale pixel values

    # Make predictions
    age_pred, gender_pred = model.predict(x)

    # Post-process predictions
    age = age_pred[0][0]
    gender = 1 if gender_pred[0][0] >= 0.5 else 0  # Convert to 0 or 1 based on threshold

    return age, gender

# Example usage (replace with a path to your image)
image_path = 'your img path'
predicted_age, predicted_gender = predict_age_and_gender(image_path)
print(f"Predicted Age: {predicted_age:.2f}")
print(f"Predicted Gender: {'Male' if predicted_gender == 0 else 'Female'}")
