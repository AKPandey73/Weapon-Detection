from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the YOLO model (assuming YOLOv8 is being used)
model = YOLO('best.pt')  

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image with the YOLO model
        results = model(filepath)

        # Save the results
        for result in results:
            result_image = result.plot()  # This returns the image with annotations
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, result_image)

        return redirect(url_for('display_image', filename='result_' + filename))

@app.route('/capture')
def capture():
    cap = cv2.VideoCapture(0)  # Open the default camera
    ret, frame = cap.read()

    if ret:
        filename = 'captured_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)

        # Process the image with the YOLO model
        results = model(filepath)

        # Save the results
        for result in results:
            result_image = result.plot()  # This returns the image with annotations
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, result_image)

        cap.release()
        return redirect(url_for('display_image', filename='result_' + filename))

    cap.release()
    return 'Failed to capture image from the camera.'

@app.route('/display/<filename>')
def display_image(filename):
    file_url = url_for('static', filename='uploads/' + filename)
    return render_template('display.html', file_url=file_url)

if __name__ == '__main__':
    app.run(debug=True)
