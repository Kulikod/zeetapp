from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask import send_file

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the model
model = load_model("plant_model.h5", compile=False)

# Load the labels
class_names = open("plant_labels.txt", "r").readlines()

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(file_path):
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        class_name, confidence_score = process_image(file_path)
        result = {
            'class_name': class_name[2:],
            'confidence_score': str(confidence_score),
            'image_path': file_path
        }
        return jsonify(result)
    
@app.route('/proxy')
def proxy():
    url = request.args.get('url')
    return send_file(url)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')