# # src/test_recommend.py

# from modeling.recommend import FashionRecommender

# if __name__ == "__main__":
#     csv_path = r"C:\Users\ranit\demo-project\data\myntradataset\styles.csv"
#     image_folder = r"C:\Users\ranit\demo-project\data\myntradataset\images"

#     recommender = FashionRecommender(csv_path=csv_path, image_folder=image_folder)

#     recommender.extract_features()  # Extract ResNet embeddings

#     test_item_id = 49926  # Example ID, replace with any from your CSV
#     similar_items = recommender.get_similar_items(test_item_id, n=5)

#     print(similar_items)
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modeling.recommend import FashionRecommender



recommender = FashionRecommender()
recommender.extract_features()
print(recommender.get_similar_items(49926, n=5))
