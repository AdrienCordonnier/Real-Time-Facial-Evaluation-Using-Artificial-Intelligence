import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

# Preprocess the data in batches
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 and normalize to [0, 1]
    label = tf.cast(label, tf.float32)
    return image, label

def load_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Create testing dataset
batch_size = 32
test_dataset = load_dataset(X_test, y_test, batch_size)

# Load the best model
model = tf.keras.models.load_model('D:/adrie/Téléchargements/archive/best_model.keras') #change path

# Evaluate the model
test_loss, test_mae = model.evaluate(test_dataset)
print(f'Test MAE: {test_mae}')

# Load training history
history = np.load('D:/adrie/Téléchargements/archive/training_history.npy', allow_pickle=True).item() #change path

# Optional: Visualize training progress
# Plot training & validation loss values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.plot(history['mean_absolute_error'])
plt.plot(history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
