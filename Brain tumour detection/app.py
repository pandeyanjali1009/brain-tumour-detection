"""Brain Tumor Detection System - Gradio web app (YOLOv11-seg)

Run:  python app.py
Colab me: weights ka path WEIGHTS env var ya neeche default change karo.
"""
import os

import cv2
import gradio as gr
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "best.pt")   # training se mili best.pt
CONF = 0.25

model = YOLO(WEIGHTS)


def detect_tumor(image):
    if image is None:
        return None, [{"Result": "Please upload an image"}]

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model.predict(image_bgr, conf=CONF)

    output_image = image
    response = []

    for result in results:
        annotated = result.plot()
        output_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                response.append({
                    "Type": model.names[cls_id],
                    "Confidence": f"{float(box.conf[0]):.2%}",
                    "BBox": [round(x) for x in box.xyxy[0].tolist()],
                })
        else:
            response.append({"Result": "No tumors detected"})

    return output_image, response


examples = [p for p in ["tumorneuro-1/test/images/Image_1.jpg",
                        "tumorneuro-1/test/images/Image_2.jpg"] if os.path.exists(p)]

interface = gr.Interface(
    fn=detect_tumor,
    inputs=gr.Image(label="Upload Brain Scan"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.JSON(label="Detection Details"),
    ],
    title="Brain Tumor Detection System",
    description="Upload MRI scan to detect tumors (Glioma, Meningioma, Pituitary)",
    examples=[[p] for p in examples] or None,
)

if __name__ == "__main__":
    interface.launch(debug=True, share=True)
