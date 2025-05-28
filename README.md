# Tanmay's Blueprint Object Detection API 🏗️🔍

This project detects **doors** and **windows** in blueprint images using a YOLOv5n model trained on custom annotated construction plans. The API is built with **Flask** and deployed on **Render**. It accepts a blueprint image and returns a JSON response with detected objects and their bounding box coordinates.

## 🚀 Features

- Object Detection using **YOLOv5n**
- Custom annotations (doors and windows) via `labelImg`
- Flask API to upload images and get JSON results
- Deployed using [Render](https://render.com/)
- JSON response with object type and bounding box coordinates

---

## 📁 Dataset Preparation

- Images were blueprints of construction sites
- Annotated using `labelImg` with two classes: `door` and `window`
- Split into:
  - `80%` training data
  - `20%` validation data

---

## 🧠 Model Training

- Model: `YOLOv5n` (nano variant for speed)
- Framework: [Ultralytics YOLOv5n]
## Render Deployed link- https://render-palcode.onrender.com/
## Result of curl (test locally)
curl -X POST  http://127.0.0.1:10000/detect -F "image=@C:\Users\tanma\OneDrive\Desktop\ex_image.png" # This is a local image file
{"detections":[{"bbox":[1169.9,64.44,100.56,100.85],"confidence":0.89,"label":"door"},{"bbox":[1590.2,14.85,116.18,113.63],"confidence":0.85,"label":"door"},{"bbox":[777.27,93.91,177.9,139.37],"confidence":0.84,"label":"door"},{"bbox":[403.82,722.76,100.88,103.62],"confidence":0.81,"label":"door"},{"bbox":[1033.51,298.07,45.26,164.22],"confidence":0.81,"label":"window"},{"bbox":[1362.66,436.74,101.75,102.46],"confidence":0.81,"label":"door"},{"bbox":[802.83,766.43,75.91,78.95],"confidence":0.81,"label":"door"},{"bbox":[1438.36,936.62,109.84,107.88],"confidence":0.8,"label":"door"},{"bbox":[1706.45,250.49,88.47,111.57],"confidence":0.8,"label":"door"},{"bbox":[295.0,756.24,98.3,109.94],"confidence":0.8,"label":"door"},{"bbox":[397.25,449.66,76.53,83.49],"confidence":0.78,"label":"door"},{"bbox":[484.67,455.53,99.41,114.35],"confidence":0.78,"label":"door"},{"bbox":[1058.64,75.59,102.93,125.82],"confidence":0.77,"label":"door"},{"bbox":[1033.95,809.17,48.64,226.54],"confidence":0.75,"label":"window"},{"bbox":[1432.37,1138.39,116.15,110.23],"confidence":0.73,"label":"door"},{"bbox":[1034.19,551.35,49.97,164.85],"confidence":0.72,"label":"window"},{"bbox":[394.02,279.27,85.57,80.99],"confidence":0.65,"label":"door"},{"bbox":[1065.2,0.44,91.44,70.46],"confidence":0.56,"label":"door"},{"bbox":[390.51,870.7,23.05,134.95],"confidence":0.32,"label":"window"},{"bbox":[1032.78,548.56,53.24,263.63],"confidence":0.29,"label":"window"}]}
