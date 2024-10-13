import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'D:/adrie/Téléchargements/archive/scut_fbp5500-cmprsd.npz' #change the path
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

# Create training and testing datasets
batch_size = 32
train_dataset = load_dataset(X_train, y_train, batch_size)
test_dataset = load_dataset(X_test, y_test, batch_size)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(350, 350, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

# Set up callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('D:/adrie/Téléchargements/archive/best_model.keras', monitor='val_loss', save_best_only=True, mode='min') #change the path
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=30,  # Reduced epochs to 20
    callbacks=[checkpoint, early_stop]
)

# Save training history
np.save('D:/adrie/Téléchargements/archive/training_history.npy', history.history)
