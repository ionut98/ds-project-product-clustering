import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def download_and_preprocess(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Eroare la: {image_url} → {e}")
        return None

# 1. Citește CSV-ul
df = pd.read_csv("ds_products.csv", sep=";")

# 2. Împarte în emag / tei
df_emag = df[df['Source'].str.lower().str.contains('emag')]
df_tei = df[df['Source'].str.lower().str.contains('tei')]

# 3. Prelucrează imaginile
def get_images(df, folder_prefix):
    images = []
    names = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row['Image']
        name = row.get('Name', f"{folder_prefix}_{idx}")
        img_tensor = download_and_preprocess(img_url)
        if img_tensor is not None:
            images.append(img_tensor)
            names.append(name)
    return images, names

print("📥 Descarc imagini eMAG...")
emag_imgs, emag_names = get_images(df_emag, "emag")

print("📥 Descarc imagini Farmacia Tei...")
tei_imgs, tei_names = get_images(df_tei, "tei")

# 4. Extrage embeddings
emag_features = torch.cat([model.encode_image(img) for img in tqdm(emag_imgs)])
tei_features = torch.cat([model.encode_image(img) for img in tqdm(tei_imgs)])

emag_features /= emag_features.norm(dim=-1, keepdim=True)
tei_features /= tei_features.norm(dim=-1, keepdim=True)

# 5. Similaritate
similarities = cosine_similarity(emag_features.cpu().numpy(), tei_features.cpu().numpy())

threshold = 0.92
print("\n🔍 POTRIVIRI posibile:")
for i, emag_name in enumerate(emag_names):
    for j, tei_name in enumerate(tei_names):
        score = similarities[i][j]
        if score > threshold:
            print(f"✅ {emag_name} ↔ {tei_name} (similaritate: {score:.3f})")


with open("product_matches.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["emag_name", "emag_image", "tei_name", "tei_image", "similarity_score"])
    for i, emag_name in enumerate(emag_names):
        for j, tei_name in enumerate(tei_names):
            score = similarities[i][j]
            if score > threshold:
                emag_img = df_emag.iloc[i]['image_url']
                tei_img = df_tei.iloc[j]['image_url']
                writer.writerow([emag_name, emag_img, tei_name, tei_img, round(score, 4)])