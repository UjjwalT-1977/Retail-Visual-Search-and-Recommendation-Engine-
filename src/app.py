import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

# ------------------------------------------------------------------
# 1. Page Configuration & Paths
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Visual Search & Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR      = PROJECT_ROOT / "data" / "raw" / "images"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
MODELS_DIR   = PROJECT_ROOT / "models"
INDEX_PATH   = PROJECT_ROOT / "artifacts" / "index" / "product_index.faiss"
DB_PATH      = PROJECT_ROOT / "artifacts" / "metadata" / "catalog.db"
EXP_CSV_PATH = PROJECT_ROOT / "reports" / "experiment_comparison.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2. Model Architectures (E3 Fine-Tuned ResNet-18 + E4 Triplet Head)
# ------------------------------------------------------------------
class ResNet18RetailClassifier(nn.Module):
    def __init__(self, num_classes=51, dropout_p=0.2):
        super().__init__()
        base = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten(start_dim=1))
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x):
        return nn.functional.normalize(self.backbone(x), p=2, dim=1)


class ResidualMetricHead(nn.Module):
    def __init__(self, dim=512, hidden_dim=256, num_classes=51, dropout_p=0.2):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, dim),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, h_in):
        h_refined = nn.functional.normalize(h_in + 0.3 * self.adapter(h_in), p=2, dim=1)
        return self.classifier(h_refined), h_refined


# ------------------------------------------------------------------
# 3. Transparent Rule-Based Complementary Category Map
# ------------------------------------------------------------------
COMPLEMENTARY_RULES = {
    "Tshirts":      ["Jeans", "Shorts", "Casual Shoes", "Watches", "Sunglasses", "Backpacks"],
    "Shirts":       ["Trousers", "Jeans", "Formal Shoes", "Belts", "Watches"],
    "Tops":         ["Jeans", "Flats", "Heels", "Handbags", "Watches", "Sunglasses"],
    "Kurtas":       ["Sandals", "Flats", "Heels", "Handbags", "Watches", "Earrings"],
    "Dresses":      ["Heels", "Flats", "Handbags", "Clutches", "Watches", "Sunglasses"],
    "Jeans":        ["Tshirts", "Shirts", "Tops", "Casual Shoes", "Belts", "Watches"],
    "Trousers":     ["Shirts", "Tshirts", "Formal Shoes", "Belts", "Watches"],
    "Shorts":       ["Tshirts", "Flip Flops", "Casual Shoes", "Sports Shoes", "Sunglasses"],
    "Casual Shoes": ["Jeans", "Tshirts", "Shirts", "Watches", "Backpacks", "Belts"],
    "Sports Shoes": ["Tshirts", "Shorts", "Track Pants", "Watches", "Backpacks", "Socks"],
    "Formal Shoes": ["Trousers", "Shirts", "Belts", "Watches", "Wallets"],
    "Heels":        ["Dresses", "Tops", "Kurtas", "Handbags", "Clutches"],
    "Flats":        ["Kurtas", "Tops", "Jeans", "Handbags", "Watches"],
    "Sandals":      ["Kurtas", "Tshirts", "Jeans", "Watches", "Sunglasses"],
    "Watches":      ["Shirts", "Tshirts", "Casual Shoes", "Sunglasses", "Wallets"],
    "Handbags":     ["Dresses", "Tops", "Kurtas", "Heels", "Watches", "Sunglasses"],
    "Backpacks":    ["Tshirts", "Jeans", "Casual Shoes", "Sports Shoes", "Watches"],
    "Sunglasses":   ["Tshirts", "Shirts", "Watches", "Casual Shoes", "Handbags"],
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------
# 4. Cached Loading of Models, FAISS Index, and Label Mapping
# ------------------------------------------------------------------
@st.cache_resource
def load_system_assets():
    with open(SPLITS_DIR / "label_mapping.json") as f:
        mapping = json.load(f)
    idx_to_label = {int(k): v for k, v in mapping["idx_to_label"].items()}
    num_classes = len(idx_to_label)

    # Load E3 Fine-Tuned ResNet-18 Backbone
    backbone_model = ResNet18RetailClassifier(num_classes=num_classes).to(DEVICE)
    ckpt_e3 = torch.load(MODELS_DIR / "e3_finetuned_resnet18.pth", map_location=DEVICE)
    backbone_model.load_state_dict(ckpt_e3["model_state_dict"])
    backbone_model.eval()

    # Load E4 Batch-Hard Triplet Metric Projection Head
    metric_head = ResidualMetricHead(dim=512, hidden_dim=256, num_classes=num_classes).to(DEVICE)
    metric_head.load_state_dict(torch.load(MODELS_DIR / "e4_metric_triplet_head.pth", map_location=DEVICE))
    metric_head.eval()

    # Load FAISS IndexFlatIP (20,906 x 512)
    faiss_index = faiss.read_index(str(INDEX_PATH))
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    return backbone_model, metric_head, faiss_index, idx_to_label, test_df


def query_sql_by_faiss_ids(faiss_ids):
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" for _ in faiss_ids)
    sql = f"SELECT faiss_idx, id, productDisplayName, articleType, baseColour, gender, usage, season FROM products WHERE faiss_idx IN ({placeholders})"
    df = pd.read_sql_query(sql, conn, params=[int(i) for i in faiss_ids])
    conn.close()
    # Preserve exact FAISS rank order
    df = df.set_index("faiss_idx").loc[[int(i) for i in faiss_ids]].reset_index()
    return df


def query_complementary_sql(pred_category, inferred_gender=None, limit=4):
    comp_cats = COMPLEMENTARY_RULES.get(pred_category, ["Watches", "Sunglasses", "Casual Shoes", "Backpacks"])
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" for _ in comp_cats)
    if inferred_gender and inferred_gender in {"Men", "Women", "Boys", "Girls", "Unisex"}:
        sql = f"""
            SELECT id, productDisplayName, articleType, baseColour, gender, usage
            FROM products
            WHERE articleType IN ({placeholders}) AND (gender = ? OR gender = 'Unisex')
            ORDER BY RANDOM() LIMIT ?
        """
        params = comp_cats + [inferred_gender, limit]
    else:
        sql = f"""
            SELECT id, productDisplayName, articleType, baseColour, gender, usage
            FROM products
            WHERE articleType IN ({placeholders})
            ORDER BY RANDOM() LIMIT ?
        """
        params = comp_cats + [limit]
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df, comp_cats


@torch.no_grad()
def run_inference(pil_img, backbone_model, metric_head, faiss_index, idx_to_label, top_k=5):
    t0 = time.perf_counter()
    img_rgb = pil_img.convert("RGB")
    x = EVAL_TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)

    h_e3 = backbone_model.extract_features(x)      # [1, 512]
    logits, h_e4 = metric_head(h_e3)               # [1, 51], [1, 512]
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    query_vec = h_e4.cpu().numpy().astype(np.float32)

    t_search0 = time.perf_counter()
    scores, indices = faiss_index.search(query_vec, top_k)
    faiss_ms = (time.perf_counter() - t_search0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    top3_idx = np.argsort(-probs)[:3]
    top3_preds = [(idx_to_label[int(i)], float(probs[i])) for i in top3_idx]

    return top3_preds, scores[0], indices[0], faiss_ms, total_ms


# ------------------------------------------------------------------
# 5. Streamlit UI Layout
# ------------------------------------------------------------------
st.title("🛍️ Retail Visual Search & Recommendation Engine")
st.caption("ResNet-18 (Layer-4 Fine-Tuned) + Batch-Hard Cosine Triplet Head + FAISS Vector DB + SQLite Catalog Metadata")

backbone_model, metric_head, faiss_index, idx_to_label, test_df = load_system_assets()

# Sidebar Controls (Great for Live Viva Debugging / Threshold Change!)
st.sidebar.header("⚙️ Search & Inference Controls")
top_k = st.sidebar.slider("Top-K Visually Similar Items", min_value=3, max_value=10, value=5)
conf_threshold = st.sidebar.slider("Confidence Warning Threshold (%)", min_value=20, max_value=90, value=50)
show_complements = st.sidebar.checkbox("Show Rule-Based Complementary Outfits", value=True)
st.sidebar.markdown("---")
st.sidebar.info(f"**Active Compute:** `{str(DEVICE).upper()}`\n\n**Indexed Catalog:** `{faiss_index.ntotal:,}` unique SKUs\n\n**Categories:** `{len(idx_to_label)}`")

# Input Source: File Upload OR Random Unseen Test Sample
col_up, col_sample = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader("Upload a retail product image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
with col_sample:
    st.write("Or test immediately on an unseen held-out image:")
    if st.button("🎲 Load Random Unseen Test Image", use_container_width=True):
        sample_row = test_df.sample(1).iloc[0]
        st.session_state["sample_img_path"] = str(IMG_DIR / f"{sample_row['id']}.jpg")
        st.session_state["sample_true_label"] = f"{sample_row['articleType']} (ID: {sample_row['id']})"

query_img = None
query_source_caption = ""

if uploaded_file is not None:
    try:
        query_img = Image.open(uploaded_file)
        query_img.verify()  # Validate file integrity
        uploaded_file.seek(0)
        query_img = Image.open(uploaded_file).convert("RGB")
        query_source_caption = f"Uploaded Image: {uploaded_file.name}"
    except (UnidentifiedImageError, OSError, Exception) as e:
        st.error(f"Invalid or corrupt image file uploaded. Please upload a valid JPG/PNG image. (Details: {e})")
elif "sample_img_path" in st.session_state:
    query_img = Image.open(st.session_state["sample_img_path"]).convert("RGB")
    query_source_caption = f"Held-Out Test Sample — True Label: {st.session_state['sample_true_label']}"

# Run Inference & Display Results
if query_img is not None:
    top3_preds, sim_scores, faiss_ids, faiss_ms, total_ms = run_inference(
        query_img, backbone_model, metric_head, faiss_index, idx_to_label, top_k=top_k
    )
    pred_cat, pred_conf = top3_preds[0]

    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(query_img, caption=query_source_caption, width=260)
    with c2:
        st.subheader(f"Predicted Category: `{pred_cat}` ({pred_conf*100:.1f}%)")
        if pred_conf * 100 < conf_threshold:
            st.warning(
                f"⚠️ **Low Confidence ({pred_conf*100:.1f}% < {conf_threshold}%):** "
                "This image may be out-of-distribution (non-catalog item, cluttered background, or rare subcategory). "
                "Treat retrieved results as approximate visual matches."
            )
        st.write("**Top-3 Category Confidence Distribution:**")
        conf_df = pd.DataFrame({"Category": [p[0] for p in top3_preds], "Confidence (%)": [round(p[1]*100, 2) for p in top3_preds]})
        st.bar_chart(conf_df.set_index("Category"))

        m1, m2, m3 = st.columns(3)
        m1.metric("FAISS Search Latency", f"{faiss_ms:.3f} ms")
        m2.metric("End-to-End Inference", f"{total_ms:.1f} ms")
        m3.metric("Top-1 Cosine Similarity", f"{sim_scores[0]:.4f}")

    # ------------------------------------------------------------------
    # Section A: Top-K Visually Similar Products (FAISS + SQLite Join)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"🔍 Top-{top_k} Visually Similar Catalog Products (FAISS Metric Retrieval)")
    retrieved_df = query_sql_by_faiss_ids(faiss_ids)

    cols = st.columns(min(top_k, 5))
    for i, row in retrieved_df.iterrows():
        with cols[i % 5]:
            img_path = IMG_DIR / f"{row['id']}.jpg"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            st.markdown(f"**Rank #{i+1}** · `cos = {sim_scores[i]:.3f}`")
            st.caption(f"**{row['articleType']}** ({row['baseColour']})\n\n{row['productDisplayName']}")

    # ------------------------------------------------------------------
    # Section B: Transparent Rule-Based Complementary Suggestions
    # ------------------------------------------------------------------
    if show_complements:
        st.markdown("---")
        inferred_gender = retrieved_df["gender"].mode()[0] if len(retrieved_df) > 0 else None
        comp_df, comp_cats = query_complementary_sql(pred_cat, inferred_gender=inferred_gender, limit=4)
        st.subheader("🧩 Complementary Outfit Suggestions *(Transparent Rule-Based Engine)*")
        st.caption(
            f"Note: Visual search above finds **substitutes** (`{pred_cat}`). Because Kaggle `styles.csv` contains no user co-purchase logs, "
            f"complementary items below are queried via transparent SQL rules matching inferred gender (`{inferred_gender}`) and compatible categories: `{', '.join(comp_cats[:4])}`."
        )
        comp_cols = st.columns(4)
        for j, crow in comp_df.iterrows():
            with comp_cols[j % 4]:
                c_img_path = IMG_DIR / f"{crow['id']}.jpg"
                if c_img_path.exists():
                    st.image(str(c_img_path), use_container_width=True)
                st.markdown(f"**{crow['articleType']}** ({crow['baseColour']})")
                st.caption(f"{crow['productDisplayName']} · {crow['gender']} / {crow['usage']}")

# ------------------------------------------------------------------
# 6. Model Card, Ablation Results & Responsible AI Limitations
# ------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 View Controlled Ablation Benchmarks & Responsible AI Model Card"):
    if EXP_CSV_PATH.exists():
        st.write("**Measured Phase I & Phase II Experiment Log (`reports/experiment_comparison.csv`):**")
        st.dataframe(pd.read_csv(EXP_CSV_PATH), use_container_width=True)
    st.markdown(
        """
        **Responsible Use & Limitations Statement:**
        - **Intended Use:** Retail catalog search, automated SKU category tagging across 51 fashion categories, and visual substitute discovery.
        - **Non-Intended Use:** Not designed for human biometric identification, facial recognition, surveillance, or non-fashion object detection.
        - **Known Limitations:** Trained on clean e-commerce studio images (Kaggle Fashion Product Images Small); real-world phone photos with cluttered backgrounds or extreme lighting may trigger lower confidence scores. Retrieval relevance is evaluated via category-level `articleType` matching, and complementary suggestions are explicitly rule-based due to the absence of user co-purchase interaction logs.
        """
    )