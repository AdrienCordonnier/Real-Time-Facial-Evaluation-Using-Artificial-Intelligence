from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import random
import cv2

# Define values specific to our dataset
MEAN, SCALE = 2.98211364, 0.68779532
# If labels were scaled during training, reverse the scaling here
# Assuming the scaler used during training was saved as 'scaler.pkl'
scaler = StandardScaler()
scaler.mean_ = np.array([MEAN])  # Replace with the mean used during training
scaler.scale_ = np.array([SCALE])  # Replace with the scale used during training

# Load the best model
MODEL = tf.keras.models.load_model('best_model.keras')

# Function to calculate the score
def calculate_score(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (350, 350))           # Resize to the model's input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0                           # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)           # Add batch dimension

    # Predict the beauty score
    predicted_score = MODEL.predict(image)

    # Reverse the scaling
    predicted_score_unscaled = scaler.inverse_transform(predicted_score)

    return predicted_score_unscaled[0][0]

#print(Calculate_score("face_capture.jpg"))

###--------------------------------------------------------------------------------------

import random

import random

face_ratings = {
    1: ["Your face could make a scarecrow weep.",
        "Your face is proof that nature has a cruel sense of humor.",
        "Your face is like a train wreck in slow motion.",
        "Your face is a visual assault on my eyeballs.",
        "Your face is a masterpiece... of disaster.",
        "Someone's face is a hot mess.",
        "Someone is a walking advertisement for plastic surgery."],
    2: ["Your face is a 'before' picture that never got its 'after.'",
        "The only thing your face is good for is a Halloween mask.",
        "Your face is like a train wreck in slow motion.",
        "Your face is a visual assault on my eyeballs.",
        "Your face is a 'before' picture that never got its 'after.'",
        "Your face is a train wreck.",
        "Your face is a disappointment in HD."],
    3: ["Your face is forgettable, like elevator music.",
        "Your face is the reason why birth control was invented.",
        "Your face is the visual equivalent of a participation award.",
        "Your face is as interesting as watching paint dry.",
        "Someone's face is a blank canvas...without the potential for art.",
        "Your face is as generic as a store brand cereal.",
        "Your face is the 'meh' of faces."],
    4: ["Your face is as beautiful as a sunrise...in Siberia.",
        "Your face is like a rare Pokemon—hard to find, but not worth it.",
        "Your face is like a math problem: It has its flaws, but it is solid.",
        "Your face is like a sunset...in a post-apocalyptic wasteland.",
        "Your face is the reason why some animals eat their young.",
        "Your face is like a book: I would rather wait for the movie adaptation.",
        "Your face is the kind of face that could stop a clock."],
    5: ["Your face is so beautiful, it could make a sculpture weep.",
        "Your face is like a summer breeze—refreshing and delightful.",
        "If your face were a song, it would be a chart-topper.",
        "Your face is the kind of face that makes angels jealous.",
        "Your face is like a shooting star—rare, breathtaking, and unforgettable.",
        "Your face is so stunning, it's like staring into the sun...without the risk of blindness.",
        "Your face is a masterpiece. Don't let anyone tell you otherwise."]
}


# Choose a random comment for a given rating, ensuring it's different from the previous one
def random_comment(rating, last_comment=None):
    comments = face_ratings[rating]
    new_comment = random.choice(comments)
    while new_comment == last_comment:
        new_comment = random.choice(comments)
    return new_comment
