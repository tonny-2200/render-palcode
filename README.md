# Tanmay's Blueprint Object Detection API 🏗️🔍

This project detects **doors** and **windows** in blueprint images using a YOLOv8n model trained on custom annotated construction plans. The API is built with **Flask** and deployed on **Render**. It accepts a blueprint image and returns a JSON response with detected objects and their bounding box coordinates.

## 🚀 Features

- Object Detection using **YOLOv8n**
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

- Model: `YOLOv8n` (nano variant for speed)
- Framework: [Ultralytics YOLOv8]
