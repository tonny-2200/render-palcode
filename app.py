from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch
import numpy as np
import gc

app = Flask(__name__)

# Load model once, on CPU
model = YOLO("model.pt")  # Assume small size model
model.to('cpu')  # Ensure it's on CPU explicitly

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Blueprint Object Detection API"}), 200

@app.route("/detect", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    try:
        file_bytes = file.read()
        with Image.open(io.BytesIO(file_bytes)) as img:
            image = img.convert("RGB")
            image_np = np.array(image)

        # Run YOLOv8 prediction with safe flags
        with torch.no_grad():
            results = model.predict(
                image_np,
                device='cpu',
                save=False,
                save_txt=False,
                save_crop=False,
                verbose=False
            )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": [round(v, 2) for v in bbox]
                })

        # Free memory
        del file_bytes, image, image_np, results
        gc.collect()
        torch.cuda.empty_cache()

        if not detections:
            return jsonify({"message": "No objects found"}), 200

        return jsonify({"detections": detections}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
