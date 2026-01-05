# src/modelling/recommend.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FashionRecommender:
    def __init__(self, csv_path=None, image_folder=None, device=None):
        # Paths
        self.csv_path = csv_path or r"C:\Users\ranit\demo-project\data\myntradataset\styles.csv"
        self.image_folder = image_folder or r"C:\Users\ranit\demo-project\data\myntradataset\images"

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(self.csv_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f)

        self.data = pd.read_csv(self.csv_path, engine="python", on_bad_lines="skip")
        print(f"Total lines in CSV: {total_lines}, Rows loaded: {len(self.data)}, Rows skipped: {total_lines - len(self.data)}")
        print(f"✅ Using device: {self.device}")

        # ResNet model (feature extractor)
        base_model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # Embeddings placeholder
        self.embeddings = None

    def extract_features(self, save_path="features.npy"):
        """
        Extract features for all dataset images OR load precomputed features if exists.
        """
        if os.path.exists(save_path):
            print(f"✅ Loading precomputed features from {save_path}")
            self.embeddings = np.load(save_path)
            return

        print("🔍 Extracting features for dataset images (this may take a while)...")
        features = []
        valid_indices = []

        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.image_folder, f"{row['id']}.jpg")
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().cpu().numpy()
                features.append(emb)
                valid_indices.append(idx)
            except:
                continue

        self.embeddings = np.array(features)
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        np.save(save_path, self.embeddings)
        print(f"✅ Features extracted and saved: {self.embeddings.shape}")

    def get_similar_items(self, item_id, n=5):
        """
        Return top-n visually similar items based on cosine similarity.
        """
        if self.embeddings is None:
            raise ValueError("You must call extract_features() first!")

        idx_list = self.data[self.data["id"] == item_id].index
        if len(idx_list) == 0:
            raise ValueError(f"Item ID {item_id} not found!")

        target_idx = idx_list[0]
        target_emb = self.embeddings[target_idx].reshape(1, -1)
        scores = cosine_similarity(target_emb, self.embeddings)[0]
        top_indices = scores.argsort()[::-1][1 : n + 1]  # exclude self

        return self.data.iloc[top_indices][
            ["id", "masterCategory", "subCategory", "articleType", "baseColour"]
        ]
