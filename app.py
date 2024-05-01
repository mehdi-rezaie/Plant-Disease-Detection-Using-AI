from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('C:/Users/Mehdi OA/Desktop/PlantDiseaseDetection/model/model.h5')

# Define the input image dimensions
input_shape = (256, 256)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty file'})

    # Read and preprocess the uploaded image
    image = Image.open(file)
    image = image.resize(input_shape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Get the selected plant type
    plant_type = request.form.get('plantType')

    # Perform the prediction based on the plant type
    if plant_type == 'apple':
        # Perform prediction for apple
        result = model.predict(image)

        # Process the result and determine the output message
        if result[0][0] < 0.1:
            output = 'This is an apple plant and it is sick, It has a Disease'
        else:
            output = 'This is an apple plant and it is healthy and has no disease'

        return jsonify({'output': output})

    elif plant_type == 'grape':
        # Perform prediction for grape
        result = model.predict(image)

        # Process the result and determine the output message
        if result[0][0] > 0.1:
            output = 'This is a grape plant and it is sick, It has a Disease'
        else:
            output = 'This is an grape plant and it is healthy and has no disease'

        return jsonify({'output': output})

    elif plant_type == 'potato':
        # Perform prediction for potato
        result = model.predict(image)
        
        # Process the result and determine the output message
        if result[0][0] > 0.1:
            output = 'This is a potato plant and it is sick, It has a Disease'
        else:
            output = 'This is an potato plant and it is healthy and has no disease'

        return jsonify({'output': output})

    else:
        return jsonify({'error': 'Invalid plant type'})



@app.route('/result')
def result():
    output = request.args.get('output')
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         # Get form values
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm-password']

#         # Validate form fields
#         if not username or not email or not password or not confirm_password:
#             return 'Please fill in all fields'

#         if password != confirm_password:
#             return 'Passwords do not match'

#         # Perform registration process
#         # Add your code here to handle the registration logic, such as storing the user in a database

#         # Redirect to login page or perform other actions after successful registration
#         return redirect('/login')

#     return render_template('register.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# if __name__ == '__main__':
#     app.run(debug=True)
