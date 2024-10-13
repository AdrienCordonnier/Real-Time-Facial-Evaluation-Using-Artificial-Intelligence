import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the best model
model = tf.keras.models.load_model('D:/adrie/Téléchargements/archive/best_model.keras') #change path

# Load the dataset
data_path = 'D:/adrie/Téléchargements/archive/scut_fbp5500-cmprsd.npz' #change path
data = np.load(data_path)
X = data['X']
y = data['y']
X.shape, y.shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the labels (optional)
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
mean = scaler.mean_
scale = scaler.scale_
print(mean)
print(scale)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (350, 350))  # Resize to the model's input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Path to the image you want to predict
image_path = 'C:/image'  # Replace with your image path

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Predict the beauty score
predicted_score = model.predict(preprocessed_image)

# If labels were scaled during training, reverse the scaling here
# Assuming the scaler used during training was saved as 'scaler.pkl'
scaler = StandardScaler()
scaler.mean_ = np.array([2.98211364])  # Replace with the mean used during training
scaler.scale_ = np.array([0.68779532])  # Replace with the scale used during training

# Reverse the scaling
predicted_score_unscaled = scaler.inverse_transform(predicted_score)

print(f'Predicted Beauty Score (Unscaled): {predicted_score_unscaled[0][0]}')