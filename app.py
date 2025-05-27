from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model once on startup
model = YOLO('model.pt')

@app.route('/predict', methods=['POST'])  # Changed route to '/predict' as you used before
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Read and convert image to RGB
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Run detection
        results = model(image)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": [round(v, 2) for v in bbox]
                })

        if not detections:
            return jsonify({'message': 'No objects found'}), 200

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use PORT env var for deployment
    app.run(host="0.0.0.0", port=port, debug=True)
