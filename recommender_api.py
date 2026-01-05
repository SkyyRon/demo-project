# src/recommender_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # remove this
from src.modeling.recommend import FashionRecommender


# Input model
class RecommendationRequest(BaseModel):
    item_id: int
    n: int = 5


# Create API
app = FastAPI(title="Fashion Recommender API")

# Initialize recommender and load precomputed features
csv_file = "C:/Users/ranit/demo-project/data/myntradataset/styles.csv"
image_folder = "C:/Users/ranit/demo-project/data/myntradataset/images"
recommender = FashionRecommender(csv_file, image_folder)
recommender.extract_features(
    save_path="features.npy"
)  # ✅ load precomputed or extract once


# Health check endpoint
@app.get("/")
def home():
    return {"message": "Fashion Recommender API is running!"}


# Recommendation endpoint
@app.post("/recommend/")
def recommend(req: RecommendationRequest):
    try:
        results = recommender.get_similar_items(req.item_id, req.n)
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
