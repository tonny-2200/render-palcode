#import necessary libraries
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Loading  model
model = YOLO(r'model.pt')  


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Run detection
        results = model(image)

        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                bbox = [x1, y1, x2 - x1, y2 - y1]  # convert to [x, y, w, h]

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": [round(val, 2) for val in bbox]
                })

        # Check if any detections were made
        if not detections:
            return jsonify({'message': 'No objects found'}), 200

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)