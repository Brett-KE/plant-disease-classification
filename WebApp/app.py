import os
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load the trained model
model = tf.keras.models.load_model('plantDisease_model.h5')

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Extract unique plant types
plant_types = list(set([cls.split("___")[0].split(" (")[0] for cls in class_names]))
plant_types.sort()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    selected_type = None

    if request.method == "POST":
        file = request.files["file"]
        selected_type = request.form.get("plant_type")

        if file and selected_type:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            # Load and preprocess image
            img = image.load_img(filepath, target_size=(256, 256))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_label = class_names[predicted_class]
            confidence = float(np.max(predictions)) * 100
            prediction = f"{predicted_label} ({confidence:.2f}% confidence)"
            img_path = filepath

    return render_template("index.html", prediction=prediction, img_path=img_path,
                           plant_types=plant_types, selected_type=selected_type)

if __name__ == "__main__":
    app.run(debug=True)
