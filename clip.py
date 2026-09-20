import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import torch
import open_clip
from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="YOLO + CLIP Food Predictor - UNRESTRICTED")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cpu")
yolo_model = YOLO("yolov8n.pt").to(device)

# Load CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.to(device).eval()


# ---------------- BASE DIR (no hardcoding) ----------------

# Priority:
#   1. Env var  FOOD_IMAGES_DIR
#   2. Folder "fdimages" sitting next to this script
#   3. ./fdimages in current working dir
BASE_DIR = os.environ.get(
    "FOOD_IMAGES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fdimages")
)

if not os.path.isdir(BASE_DIR):
    # fallback to CWD/fdimages
    alt = os.path.join(os.getcwd(), "fdimages")
    if os.path.isdir(alt):
        BASE_DIR = alt
    else:
        raise FileNotFoundError(
            f"❌ Food images folder not found.\n"
            f"Looked in: {BASE_DIR}\n"
            f"Set env var FOOD_IMAGES_DIR to override."
        )

print(f"📂 Loading food categories from: {BASE_DIR}")


# Dataset Setup
food_names = [
    f.replace("_", " ").lower()
    for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f))
]
text_prompts = [f"a photo of {food}" for food in food_names]

print(f"✅ Loaded {len(food_names)} food categories")

# Pre-encode text prompts for speed
with torch.no_grad():
    tokens = open_clip.tokenize(text_prompts).to(device)
    text_features = clip_model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)


@app.post("/predict_food")
async def predict_food(images: List[UploadFile] = File(...)):
    results_all = []

    for image in images:
        try:
            img = Image.open(image.file).convert("RGB")
            detections = yolo_model(img)

            crops = []
            for r in detections:
                for box in r.boxes.xyxy.cpu().numpy():
                    coords = list(map(int, box))
                    crops.append(img.crop(coords))

            if not crops:
                crops = [img]

            image_predictions = []
            seen_foods = {}

            for crop in crops:
                img_tensor = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img_tensor)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)

                    similarity = (img_feat @ text_features.T)[0]
                    probs, indices = similarity.topk(5)

                    for i in range(len(probs)):
                        name = food_names[int(indices[i])]
                        conf = round(float(probs[i] * 100), 2)

                        if conf > 10:
                            if name not in seen_foods or conf > seen_foods[name]:
                                seen_foods[name] = conf

            for food, score in seen_foods.items():
                image_predictions.append({
                    "food_name": food,
                    "confidence": f"{score}%"
                })

            image_predictions.sort(
                key=lambda x: float(x["confidence"].replace('%', '')),
                reverse=True
            )
            results_all.append({"image": image.filename, "predictions": image_predictions})

        except Exception as e:
            print(f"❌ Error processing image {image.filename}: {e}")

    return {"results": results_all}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)