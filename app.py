from flask import Flask, request, jsonify
from PIL import Image
import os
import io
from ultralytics import YOLO
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Define paths for the models
YOLO_MODEL_PATH = './model/training_logs/weights/best.pt'
OCR_MODEL_PATH = './model/trocr-finetuned'
OCR_PROCESSOR_PATH = 'microsoft/trocr-base-handwritten'

# Load the YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the OCR model and processor
ocr_model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_PATH)
ocr_model.to(device)
processor = TrOCRProcessor.from_pretrained(OCR_PROCESSOR_PATH, clean_up_tokenization_spaces=True)

class_id_to_label = {
    0.0: "numbers_1",
    1.0: "numbers_2",
    2.0: "numbers_3"
}

def detect_with_rotation(image_path, rotations):
    all_boxes = []
    all_angles = []
    all_images = []

    for angle in rotations:
        rotated_image = Image.open(image_path).rotate(angle)
        rotated_image_path = f"rotated_image_{angle}.jpg"
        rotated_image.save(rotated_image_path)

        results = yolo_model(rotated_image_path)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                all_boxes.extend(boxes)
                all_angles.append(angle)
                all_images.append(rotated_image)
        if len(boxes) > 0:
            return boxes, angle, rotated_image

        os.remove(rotated_image_path)

    return None, None, Image.open(image_path)

def crop_image(image, box):
    x_min, y_min, x_max, y_max = box
    return image.crop((x_min, y_min, x_max, y_max))

# Function for OCR inference
def ocr_inference(image, model, processor):
    try:
        # Convert PIL image to NumPy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        # Perform OCR
        outputs = model.generate(pixel_values)

        # Decode the output
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in OCR inference: {e}")
        return "Error during OCR inference"

@app.route('/')
def index():
    return jsonify({
        'status': {
            'code': 200,
            'message': 'Success fetching the API'
        },
        "data": None
    }), 200

@app.route('/web')
def web():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        image_path = "uploaded_image.jpg"
        file.save(image_path)
        
        rotation_angles = [0, 180, 90, -90]
        boxes, used_angle, processed_image = detect_with_rotation(image_path, rotation_angles)
        os.remove(image_path)

        if boxes:
            bounding_boxes_with_classes = []
            cropped_images = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                class_id = box.cls.tolist()[0]
                confidence = box.conf.tolist()[0]
                label = class_id_to_label.get(class_id, "Unknown")

                bounding_boxes_with_classes.append({
                    "label": label,
                    "confidence": confidence,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })

                cropped_image = crop_image(processed_image, (xmin, ymin, xmax, ymax))
                cropped_images.append(cropped_image)

            recognized_texts = [ocr_inference(cropped_image, ocr_model, processor) for cropped_image in cropped_images]

            results = []
            for box, text in zip(bounding_boxes_with_classes, recognized_texts):
                results.append({
                    "label": box['label'],
                    "bounding_box": (box['xmin'], box['ymin'], box['xmax'], box['ymax']),
                    "recognized_text": text
                })

            return jsonify({"rotation_used": used_angle, "results": results})

        else:
            return jsonify({"message": "No objects detected after trying all rotations."})

    return jsonify({"error": "Failed to process the image."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)