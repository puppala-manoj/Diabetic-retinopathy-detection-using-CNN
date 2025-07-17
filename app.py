from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import sys
app = Flask(__name__)
# Load the pre-trained model
model = load_model("model.h5")
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        # Use BytesIO to handle the file in-memory
        img_bytes = BytesIO(file.read())
        # Load the image from BytesIO
        img = image.load_img(img_bytes, target_size=(224, 224))
        # Continue with the rest of the code as before...
        # Convert the image to a numpy array, make predictions, etc.
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        # Make a prediction
        prediction = model.predict(img_array)
        # Determine the predicted class (0 or 1)
        predicted_label = "Diabetic Retinopathy" if prediction[0][0] > 0.5 else "No Issues Detected"
        # prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
        # prediction = model.predict(img_array)

        # if prediction[0][0] < 0.2:
        #     predicted_label = "No Diabetic Retinopathy"
        # elif prediction[0][0] < 0.4:
        #     predicted_label = "Mild Diabetic Retinopathy"
        # elif prediction[0][0] < 0.6:
        #     predicted_label = "Moderate Diabetic Retinopathy"
        # elif prediction[0][0] < 0.8: 
        #    predicted_label = "Severe Diabetic Retinopathy"
        # else:
        #     predicted_label = "Advanced Diabetic Retinopathy"

        return render_template('index.html', prediction=predicted_label)

    except Exception as e:
        return str(e)
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')


