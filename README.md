# Diabetic-retinopathy-detection-using-CNN
  A CNN based deep learning project to detect diabetic retinopathy from retinal images

# Problem statement 
  Diabetic Retinopathy is one of the leading causes of blindness globally, and early detection is key. Manual screening is slow and     requires expert ophthalmologists. My project aims to automate this process using a trained CNN model that can give **real-time diagnostic predictions** through a web interface.

# What this project does
- It detects the diabetic retinopathy from fundus images
- Utilizes a custom CNN built with Keras
- Preprocesses images using Gaussian filtering
- Allows users to upload images via a Flask-based web interface and detect diabetic retinopathy
  
# Tech stack
- **Frontend**: HTML, CSS (Bootstrap), Jinja2 templates
- **Backend**: Flask, Python
- **Deep Learning**: TensorFlow, Keras, OpenCV
- **Others**: NumPy, Pandas, Matplotlib

# How this project works
- pip install -r requirements.txt
- python app.py
  
- Then visit http://localhost:5000 for the output

# Model performance
- Validation Accuracy: 90.91%
- Test Accuracy: 94.00%
- Dataset: APTOS 2019


