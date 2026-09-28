"""
Streamlit UI for the Deepfake Image Classifier.

A modern, dark-themed dashboard that exposes:
  - Home / Dashboard overview
  - Dataset Management (counts, split, visualization)
  - Training (config + live progress)
  - Prediction (single image + batch)
  - Model Info (architecture, metrics)

Run with:
    streamlit run app.py
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make sure src/ is importable regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------- Page config ----------------------------- #
# Sidebar starts expanded on desktop, collapsed on mobile (users open it via
# the hamburger button in the top-left).
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on mobile, expanded on desktop
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "🛡️ Deepfake Detector — AI-powered image authenticity analysis",
    },
)

# ----------------------------- Theming / CSS ----------------------------- #
CUSTOM_CSS = """
<style>
/* ==================== BASE ==================== */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(99, 102, 241, 0.15), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(236, 72, 153, 0.12), transparent 55%),
        linear-gradient(180deg, #0b0f1a 0%, #0a0d14 100%);
    color: #e6e9f2;
}

/* Show the sidebar hamburger toggle (streamlit's native button) */
header [data-testid="stToolbar"] { visibility: visible !important; }
[data-testid="stToolbar"] { z-index: 9999 !important; }

/* Style the native header to blend with dark theme */
header[data-testid="stHeader"] {
    background: rgba(10, 13, 20, 0.92);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(8px);
}

/* Footer — hide completely */
footer { visibility: hidden; }

/* Streamlit's native MainMenu — keep but style it */
#MainMenu { visibility: visible; }

/* Hide sidebar completely — we use top nav instead */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* Headings */
h1, h2, h3, h4 { color: #f4f6fb !important; letter-spacing: -0.01em; }

/* Containers — constrain width on wide screens */
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ==================== CARDS ==================== */
.glass-card {
    background: rgba(20, 24, 38, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    transition: transform .2s ease, border-color .2s ease;
    width: 100%;
}
.glass-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.45); }

/* ==================== HERO ==================== */
.hero {
    background:
        radial-gradient(800px 400px at 0% 0%, rgba(99,102,241,0.35), transparent 60%),
        radial-gradient(700px 350px at 100% 100%, rgba(236,72,153,0.25), transparent 60%),
        linear-gradient(135deg, #111526 0%, #0b0f1a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 28px 28px 24px;
    margin-bottom: 20px;
    overflow: hidden;
}
.hero h1 {
    font-size: 2rem !important;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc 0%, #f0abfc 50%, #fcd34d 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero p { color: #c5c9d8; font-size: 0.95rem; margin-top: 8px; max-width: 780px; }
.hero .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #c7d2fe;
    font-size: 0.72rem;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ==================== STAT TILES ==================== */
.stat-tile {
    background: rgba(20, 24, 38, 0.7);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px;
    text-align: left;
    height: 100%;
    min-height: 88px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.stat-tile .label { color: #9aa3b9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.stat-tile .value { color: #f4f6fb; font-size: 1.7rem; font-weight: 700; margin-top: 2px; line-height: 1.1; }
.stat-tile.indigo   { border-top: 3px solid #6366f1; }
.stat-tile.pink     { border-top: 3px solid #ec4899; }
.stat-tile.amber    { border-top: 3px solid #f59e0b; }
.stat-tile.emerald  { border-top: 3px solid #10b981; }
.stat-tile.cyan     { border-top: 3px solid #06b6d4; }
.stat-tile.rose     { border-top: 3px solid #f43f5e; }

/* ==================== BUTTONS ==================== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35);
    width: 100%;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(99,102,241,0.5); }
.stButton > button:active { transform: translateY(0); }

/* ==================== FILE UPLOADER ==================== */
[data-testid="stFileUploader"] {
    background: rgba(20, 24, 38, 0.6);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 14px;
}

/* ==================== VERDICT BADGE ==================== */
.verdict {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: 0.05em;
}
.verdict.real {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.5);
    color: #6ee7b7;
}
.verdict.fake {
    background: linear-gradient(135deg, rgba(244,63,94,0.2), rgba(244,63,94,0.05));
    border: 1px solid rgba(244,63,94,0.5);
    color: #fda4af;
}

/* ==================== PROGRESS BARS ==================== */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #6366f1, #ec4899); }

/* ==================== TABS ==================== */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(20, 24, 38, 0.5);
    border-radius: 10px;
    padding: 7px 14px;
    color: #c5c9d8;
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25));
    border-color: rgba(99,102,241,0.55);
    color: #fff;
}

/* ==================== METRICS ==================== */
[data-testid="stMetricValue"] { color: #f4f6fb; }
[data-testid="stMetricLabel"] { color: #9aa3b9; }

/* ==================== ALERTS ==================== */
.stAlert { border-radius: 12px; }

/* ==================== CODE BLOCKS ==================== */
.stCodeBlock { border-radius: 10px; }

/* ==================== EXPANDER ==================== */
[data-testid="stExpander"] { border-radius: 12px; }

/* ==================== TABLES ==================== */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ==================== TOOLTIP / HELP ==================== */
.stTooltipIcon { color: #6366f1 !important; }

/* ==================== DOWNLOAD BUTTON ==================== */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(16,185,129,0.3);
    width: 100%;
}

/* ==================== SELECT SLIDER / NUMBER INPUT ==================== */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-size: 0.88rem;
    color: #c5c9d8;
}

/* ==================== MOBILE SIDEBAR TOGGLE HINT ==================== */
.mobile-sidebar-hint {
    display: none;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 8px 14px;
    color: #c7d2fe;
    font-size: 0.82rem;
    margin-bottom: 12px;
    text-align: center;
}

/* ==================== RESPONSIVE — TABLET (≤1024px) ==================== */
@media (max-width: 1024px) {
    .hero h1 { font-size: 1.7rem !important; }
    .hero { padding: 22px 22px 18px; }
    .stat-tile .value { font-size: 1.45rem; }
    .stat-tile { padding: 14px; min-height: 80px; }
    .glass-card { padding: 18px 18px; }
}

/* ==================== RESPONSIVE — MOBILE (≤640px) ==================== */
@media (max-width: 640px) {
    /* Block container */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* Hero */
    .hero { padding: 18px 16px 14px; border-radius: 16px; margin-bottom: 16px; }
    .hero h1 { font-size: 1.45rem !important; }
    .hero p { font-size: 0.88rem; }

    /* Stat tiles — wrap into 3-col grid */
    .stat-tile .value { font-size: 1.3rem; }
    .stat-tile { padding: 12px 10px; min-height: 72px; }
    .stat-tile .label { font-size: 0.65rem; }

    /* Glass cards */
    .glass-card { padding: 14px 14px; border-radius: 14px; }
    .glass-card:hover { transform: none; }

    /* Verdict */
    .verdict { font-size: 1.1rem; padding: 7px 14px; }

    /* Sidebar — visible hint on small screens */
    .mobile-sidebar-hint { display: block; }

    /* Buttons */
    .stButton > button { padding: 0.5rem 0.9rem; font-size: 0.9rem; }

    /* Progress bars */
    [data-testid="stProgress"] { height: 6px !important; }

    /* Metrics — make text smaller */
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.82rem; }

    /* Tables — horizontal scroll */
    .stDataFrame thead th { font-size: 0.8rem !important; }
    .stDataFrame tbody td { font-size: 0.8rem !important; }

    /* Code blocks */
    .stCodeBlock pre { font-size: 0.78rem !important; }

    /* Selectbox / number input labels */
    .stSlider label, .stSelectbox label, .stNumberInput label { font-size: 0.8rem; }
    .stSelectbox > div > div { font-size: 0.85rem; }

    /* Expander */
    [data-testid="stExpander"] { font-size: 0.88rem; }

    /* Full-width on mobile for stacked layouts */
    section[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    section[data-testid="stHorizontalBlock"] > div {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Image captions */
    [data-testid="stImage"] figcaption { font-size: 0.75rem; }
}

/* ==================== RESPONSIVE — SMALL ANDROID / iPhone SE (≤375px) ==================== */
@media (max-width: 375px) {
    .hero h1 { font-size: 1.25rem !important; }
    .stat-tile .value { font-size: 1.1rem; }
    .stat-tile .label { font-size: 0.6rem; }
    .verdict { font-size: 1rem; padding: 6px 10px; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------- Helpers ----------------------------- #
def count_images_in_dir(directory: Path, extensions=(".jpg", ".jpeg", ".png")):
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in extensions)


def get_dataset_stats():
    """Walk data/ folder and return counts for raw, train, val, test (real/fake)."""
    stats = {}
    for split in ["raw", "train", "val", "test"]:
        for cls in ["real", "fake"]:
            key = f"{split}/{cls}"
            stats[key] = count_images_in_dir(PROJECT_ROOT / "data" / split / cls)
    return stats


def check_model_exists():
    return (PROJECT_ROOT / "checkpoints" / "best_model.pth").exists()


def get_device_info():
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "CPU"
        return cuda, name
    except Exception:
        return False, "Unknown"


# ----------------------------- Auth / Admin Gate ----------------------------- #
ADMIN_PAGES = ["🏠 Home", "📊 Dataset", "🎯 Training", "ℹ️ Model Info"]
ALL_PAGES = ["🔍 Predict"] + ADMIN_PAGES

# Read admin password from secrets if available, else fall back to default
try:
    ADMIN_PASSWORD = st.secrets["admin_password"]
except (KeyError, FileNotFoundError):
    ADMIN_PASSWORD = "admin123"  # change this!

AUTH_KEY = "df_is_admin"
NAV_KEY = "df_nav_selection"
SHOW_ADMIN_KEY = "df_show_admin_panel"

if AUTH_KEY not in st.session_state:
    st.session_state[AUTH_KEY] = False

# Default to Predict page for regular users
if NAV_KEY not in st.session_state:
    st.session_state[NAV_KEY] = "🔍 Predict"

# ---- Top bar: brand + admin toggle ----
top_left, top_right = st.columns([1, 1], gap="medium")

with top_left:
    st.markdown(
        "<div style='font-size:1.8rem;font-weight:800;margin:10px 0 0;'>"
        "<span style='background:linear-gradient(90deg,#a5b4fc 0%,#f0abfc 50%,#fcd34d 100%);"
        "-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;'>"
        "🛡️ Deepfake Detector</span></div>",
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if SHOW_ADMIN_KEY not in st.session_state:
        st.session_state[SHOW_ADMIN_KEY] = False

    if st.session_state[AUTH_KEY]:
        # Already authenticated — show lock button to log out
        if st.button("🔓 Locked in • Exit admin", use_container_width=True):
            st.session_state[AUTH_KEY] = False
            st.session_state[NAV_KEY] = "🔍 Predict"
            st.session_state[SHOW_ADMIN_KEY] = False
            st.rerun()
    elif st.session_state[SHOW_ADMIN_KEY]:
        # Password input to unlock admin
        pw = st.text_input(
            "Admin password",
            type="password",
            placeholder="Enter admin password",
            key="df_admin_pw_input",
            label_visibility="collapsed",
        )
        pw_col, btn_col = st.columns([3, 1])
        with btn_col:
            if st.button("Unlock", use_container_width=True):
                if pw == ADMIN_PASSWORD:
                    st.session_state[AUTH_KEY] = True
                    st.session_state[SHOW_ADMIN_KEY] = False
                    st.session_state[NAV_KEY] = "🏠 Home"
                    st.rerun()
                else:
                    st.error("Incorrect password")
        if st.button("Cancel", use_container_width=True):
            st.session_state[SHOW_ADMIN_KEY] = False
            st.rerun()
    else:
        # Show lock button
        if st.button("🔒 Admin", use_container_width=True):
            st.session_state[SHOW_ADMIN_KEY] = True
            st.rerun()

st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:4px 0 8px;'>",
    unsafe_allow_html=True,
)

# ---- Navigation ----
# Available pages depend on auth state
available_pages = ALL_PAGES if st.session_state[AUTH_KEY] else ["🔍 Predict"]

# Show nav only if admin (or just Predict for everyone)
if st.session_state[AUTH_KEY]:
    selected = st.segmented_control(
        "Navigate",
        options=available_pages,
        default=st.session_state[NAV_KEY] if st.session_state[NAV_KEY] in available_pages else "🔍 Predict",
        label_visibility="collapsed",
        key="df_top_nav",
    )
    if selected and selected != st.session_state[NAV_KEY]:
        st.session_state[NAV_KEY] = selected
        st.rerun()
    st.markdown("&nbsp;", unsafe_allow_html=True)

nav = st.session_state[NAV_KEY]


# ----------------------------- Page: Home ----------------------------- #
def page_home():
    st.markdown(
        """
        <div class="hero">
            <span class="badge">EfficientNet-B0</span>
            <span class="badge">PyTorch</span>
            <span class="badge">Transfer Learning</span>
            <h1>Deepfake Image Classifier</h1>
            <p>Detect AI-generated and manipulated face images with a fine-tuned
            EfficientNet-B0 backbone. Manage your dataset, train the model,
            and run inference — all from one dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_dataset_stats()
    total_raw = stats["raw/real"] + stats["raw/fake"]
    total_train = stats["train/real"] + stats["train/fake"]
    total_val = stats["val/real"] + stats["val/fake"]
    total_test = stats["test/real"] + stats["test/fake"]
    total_split = total_train + total_val + total_test

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f"<div class='stat-tile indigo'><div class='label'>Raw Real</div>"
            f"<div class='value'>{stats['raw/real']:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='stat-tile rose'><div class='label'>Raw Fake</div>"
            f"<div class='value'>{stats['raw/fake']:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='stat-tile emerald'><div class='label'>Train</div>"
            f"<div class='value'>{total_train:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='stat-tile amber'><div class='label'>Val</div>"
            f"<div class='value'>{total_val:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"<div class='stat-tile cyan'><div class='label'>Test</div>"
            f"<div class='value'>{total_test:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"<div class='stat-tile pink'><div class='label'>Total Split</div>"
            f"<div class='value'>{total_split:,}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Start")
        st.markdown(
            "1. **Add data** — drop real face images into `data/raw/real/` and "
            "deepfake/AI-generated faces into `data/raw/fake/`.\n"
            "2. **Split dataset** — go to the *Dataset* tab to auto-create "
            "train/val/test (70/15/15).\n"
            "3. **Train** — open the *Training* tab, tweak hyperparameters, and "
            "launch training. The best checkpoint is saved automatically.\n"
            "4. **Predict** — once trained, head to *Predict* to classify "
            "individual images with confidence scores."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 How it works")
        st.markdown(
            "A pretrained **EfficientNet-B0** backbone (ImageNet) is fine-tuned "
            "for binary classification. The 1280-d feature vector is passed "
            "through a small MLP head (`Dropout → Linear → ReLU → Dropout → Linear`)"
            " producing a single logit. We apply `sigmoid` to get the probability "
            "of an image being **FAKE**."
        )
        st.markdown(
            "<span class='badge'>Input 224××3</span>"
            "<span class='badge'>EfficientNet-B0</span>"
            "<span class='badge'>BCE Loss</span>"
            "<span class='badge'>AdamW</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Status banner
    model_ready = check_model_exists()
    if not model_ready:
        st.warning(
            "⚠️ No trained checkpoint found. Train a model from the **Training** "
            "tab before running predictions."
        )
    else:
        st.success("✅ Trained checkpoint detected at `checkpoints/best_model.pth`.")


# ----------------------------- Page: Dataset ----------------------------- #
def page_dataset():
    st.markdown("## 📊 Dataset Management")
    st.caption("Inspect your data, run the train/val/test split, and preview samples.")

    stats = get_dataset_stats()

    # Counts table
    st.markdown("### Counts per class")
    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown("**Split**")
    header_cols[1].markdown("**Real**")
    header_cols[2].markdown("**Fake**")
    header_cols[3].markdown("**Total**")
    for split in ["raw", "train", "val", "test"]:
        r = stats[f"{split}/real"]
        f = stats[f"{split}/fake"]
        row = st.columns([1, 1, 1, 1])
        row[0].markdown(f"`{split}/`")
        row[1].markdown(f"{r:,}")
        row[2].markdown(f"{f:,}")
        row[3].markdown(f"**{r + f:,}**")

    st.markdown("---")

    # Split controls
    st.markdown("### ✂️ Split into train / val / test")
    st.caption(
        "This will copy files from `data/raw/real` and `data/raw/fake` into "
        "`data/train`, `data/val`, and `data/test` using a configurable ratio. "
        "Existing files in those folders will be overwritten."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        train_pct = st.slider("Train %", 50, 90, 70, 1, key="train_pct")
    with c2:
        val_pct = st.slider("Val %", 5, 40, 15, 1, key="val_pct")
    with c3:
        test_pct = st.slider("Test %", 5, 40, 15, 1, key="test_pct")

    if train_pct + val_pct + test_pct != 100:
        st.error(
            f"Splits must sum to 100% (currently {train_pct + val_pct + test_pct}%)."
        )
    else:
        if st.button("🚀 Run Split", type="primary"):
            with st.spinner("Splitting dataset..."):
                # Update split_dataset.py constants and run it
                import importlib
                import split_dataset
                importlib.reload(split_dataset)
                split_dataset.SPLITS = {
                    "train": train_pct / 100.0,
                    "val": val_pct / 100.0,
                    "test": test_pct / 100.0,
                }
                for c in split_dataset.CLASSES:
                    split_dataset.split_class(c)
                st.success("✅ Dataset split complete.")
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # Preview thumbnails
    st.markdown("### 🖼️ Preview samples")
    preview_split = st.selectbox("Folder", ["raw", "train", "val", "test"])
    preview_class = st.selectbox("Class", ["real", "fake"])
    preview_dir = PROJECT_ROOT / "data" / preview_split / preview_class

    if preview_dir.exists():
        images = sorted(
            [f for f in preview_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )[:12]
        if images:
            cols = st.columns(6)
            for i, img_path in enumerate(images):
                with cols[i % 6]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
        else:
            st.info(f"No images in `data/{preview_split}/{preview_class}/` yet.")
    else:
        st.info(f"`data/{preview_split}/{preview_class}/` does not exist yet.")


# ----------------------------- Page: Training ----------------------------- #
def page_training():
    st.markdown("## 🎯 Train the Model")
    st.caption("Configure hyperparameters and start a training run.")

    stats = get_dataset_stats()
    if stats["train/real"] + stats["train/fake"] == 0:
        st.error(
            "No training data found. Add images to `data/raw/real` and "
            "`data/raw/fake`, then run the split from the **Dataset** tab."
        )
        return

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs", 1, 100, 15)
        batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
    with c2:
        lr = st.select_slider(
            "Learning rate",
            options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            value=1e-4,
        )
        weight_decay = st.select_slider(
            "Weight decay",
            options=[1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-5,
        )
    with c3:
        patience = st.number_input("Early stopping patience", 1, 20, 4)
        accumulation = st.number_input("Grad accumulation steps", 1, 16, 4)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Configuration preview")
        st.code(
            f"EPOCHS             = {epochs}\n"
            f"BATCH_SIZE         = {batch_size}\n"
            f"LR                 = {lr}\n"
            f"WEIGHT_DECAY       = {weight_decay}\n"
            f"PATIENCE           = {patience}\n"
            f"ACCUMULATION_STEPS = {accumulation}\n"
            f"DEVICE             = {'cuda' if get_device_info()[0] else 'cpu'}",
            language="python",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Dataset snapshot")
        st.metric("Train images", f"{stats['train/real'] + stats['train/fake']:,}")
        st.metric("Validation images", f"{stats['val/real'] + stats['val/fake']:,}")
        st.metric("Test images", f"{stats['test/real'] + stats['test/fake']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Start Training", type="primary"):
        run_training(
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            patience=int(patience),
            accumulation=int(accumulation),
        )


def run_training(epochs, batch_size, lr, weight_decay, patience, accumulation):
    """Run a training session with live UI updates."""
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from src.model import build_model
    from src.dataset import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.markdown("### 🏃 Training progress")
    progress_bar = st.progress(0.0)
    status = st.empty()
    metric_cols = st.columns(4)
    loss_metric = metric_cols[0].empty()
    acc_metric = metric_cols[1].empty()
    f1_metric = metric_cols[2].empty()
    auc_metric = metric_cols[3].empty()
    log_box = st.expander("📜 Training log", expanded=True)
    log_area = log_box.empty()

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_area.text("\n".join(log_lines[-200:]))

    try:
        log(f"Device: {device}")
        if device == "cpu":
            log("⚠️ No GPU — training will be slow.")
        log("Loading dataloaders...")
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=batch_size, num_workers=0
        )
        log(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

        model = build_model(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        os.makedirs("checkpoints", exist_ok=True)
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                (loss / accumulation).backward()
                if (step + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # Validate
            model.eval()
            val_loss = 0.0
            preds, labs, probs = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)
                    p = torch.sigmoid(logits)
                    preds.extend((p > 0.5).float().cpu().numpy())
                    labs.extend(labels.cpu().numpy())
                    probs.extend(p.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            acc = accuracy_score(labs, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labs, preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(labs, probs)
            except ValueError:
                auc = float("nan")

            scheduler.step(val_loss)

            log(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"acc={acc:.4f} | f1={f1:.4f} | auc={auc:.4f}"
            )

            loss_metric.metric("Loss", f"{val_loss:.4f}", delta=f"train {train_loss:.4f}", delta_color="off")
            acc_metric.metric("Accuracy", f"{acc * 100:.1f}%")
            f1_metric.metric("F1", f"{f1:.4f}")
            auc_metric.metric("AUC", f"{auc:.4f}" if auc == auc else "n/a")

            progress_bar.progress(epoch / epochs)
            status.info(f"Epoch {epoch}/{epochs} complete")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    model.state_dict(), os.path.join("checkpoints", "best_model.pth")
                )
                log("  ↳ New best model saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log(f"Early stopping after {epoch} epochs.")
                    break

        st.success("✅ Training complete. Best model saved to `checkpoints/best_model.pth`.")
    except Exception as e:
        st.error(f"Training failed: {e}")
        raise


# ----------------------------- Page: Predict ----------------------------- #
def page_predict():
    st.markdown("## 🔍 Run Inference")
    st.caption("Upload one or more images to classify as REAL or FAKE.")

    if not check_model_exists():
        st.warning(
            "No trained model found. Train a model from the **Training** tab first."
        )
        return

    # Lazy import torch and model
    import torch
    from src.model import build_model
    from src.dataset import eval_transform

    @st.cache_resource
    def load_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(device)
        ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pth"
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device

    model, device = load_model()

    # Upload section
    uploaded = st.file_uploader(
        "📷 Upload a face image to analyze",
        type=["jpg", "jpeg", "png"],
        help="For best results, use a clear, front-facing image of a person.",
    )

    # No image yet — show placeholder with instructions
    if uploaded is None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;padding:40px 0'>"
            "<div style='font-size:3rem;margin-bottom:12px'>🔍</div>"
            "<div style='font-size:1.1rem;color:#9aa3b9'>"
            "Upload an image above to check if it's <b style='color:#f4f6fb'>REAL</b> "
            "or <b style='color:#f4f6fb'>AI-generated (FAKE)</b></div>"
            "<div style='margin-top:12px;font-size:0.85rem;color:#6b7280'>"
            "Supported: JPG, PNG • Max 200 MB"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Load and display image
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    # Threshold slider (admin-only note)
    if st.session_state.get(AUTH_KEY, False):
        threshold = st.slider(
            "Sensitivity threshold",
            0.05, 0.95, 0.50, 0.01,
            help="Lower = more sensitive to detecting fakes.",
        )
    else:
        threshold = 0.50

    # Analysis
    with st.spinner("Analyzing image..."):
        tensor = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            p_fake = torch.sigmoid(logit).item()
        p_real = 1.0 - p_fake
        label = "FAKE" if p_fake > threshold else "REAL"
        conf = max(p_fake, p_real) * 100

    # Result layout
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with col_result:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        verdict_class = "fake" if label == "FAKE" else "real"
        st.markdown(
            f"<div class='verdict {verdict_class}'>{label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence: {conf:.1f}%**")
        st.markdown(
            f"<div style='margin-top:8px;color:#9aa3b9;font-size:0.88rem'>"
            f"P(REAL) = {p_real:.2%} &nbsp;·&nbsp; P(FAKE) = {p_fake:.2%}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Probability bars
        real_color = "#10b981" if label == "REAL" else "#6366f1"
        fake_color = "#f43f5e" if label == "FAKE" else "#6366f1"
        c1, c2 = st.columns(2)
        c1.progress(p_real, text=f"REAL {p_real:.0%}")
        c2.progress(p_fake, text=f"FAKE {p_fake:.0%}")

    # Batch section (below main result, admin only)
    if st.session_state.get(AUTH_KEY, False):
        st.markdown("---")
        st.markdown("### 📚 Batch Analysis")
        files = st.file_uploader(
            "Upload multiple images at once",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        if files:
            import pandas as pd
            rows = []
            gallery_cols = st.columns(4)
            for i, f in enumerate(files):
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                t = eval_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pf = torch.sigmoid(model(t)).item()
                lbl = "FAKE" if pf > 0.5 else "REAL"
                rows.append({"File": f.name, "Verdict": lbl, "P(FAKE)": f"{pf:.4f}"})
                with gallery_cols[i % 4]:
                    border = "#f43f5e" if lbl == "FAKE" else "#10b981"
                    st.markdown(
                        f"<div style='border:2px solid {border};border-radius:10px;"
                        f"padding:4px;text-align:center;font-weight:700;color:{border}'>"
                        f"{lbl}<br><span style='color:#9aa3b9;font-weight:400;font-size:0.8rem'>"
                        f"{pf*100:.1f}% fake</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.image(img, use_container_width=True)
            st.markdown("#### Results")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------- Page: Model Info ----------------------------- #
def page_model_info():
    st.markdown("## ℹ️ Model Architecture & Info")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Architecture")
        st.markdown(
            "- **Backbone**: `timm` EfficientNet-B0 (ImageNet pretrained)\n"
            "- **Feature dim**: 1280\n"
            "- **Head**: `Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.4) → Linear(256→1)`\n"
            "- **Loss**: `BCEWithLogitsLoss`\n"
            "- **Optimizer**: `AdamW` (default lr=1e-4, wd=1e-5)\n"
            "- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2)\n"
            "- **Input**: 224×224 RGB, ImageNet-normalized\n"
            "- **Output**: single logit → `sigmoid` → P(FAKE)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Checkpoint")
        ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pth"
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            st.markdown(
                f"- **Path**: `checkpoints/best_model.pth`\n"
                f"- **Size**: {size_mb:.2f} MB\n"
                f"- **Modified**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt.stat().st_mtime))}"
            )
            with open(ckpt, "rb") as f:
                st.download_button(
                    "⬇️ Download checkpoint",
                    data=f,
                    file_name="best_model.pth",
                    mime="application/octet-stream",
                )
        else:
            st.info("No checkpoint found. Train a model first.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Quick command-line reference")
    st.code(
        "# Run setup (one-time)\n"
        "python setup_folders.py\n\n"
        "# After adding images to data/raw/real and data/raw/fake\n"
        "python split_dataset.py\n\n"
        "# Train\n"
        "python src/train.py\n\n"
        "# Predict on a single image\n"
        "python src/predict.py path/to/image.jpg",
        language="bash",
    )


# ----------------------------- Router ----------------------------- #
PAGES = {
    "🏠 Home": page_home,
    "📊 Dataset": page_dataset,
    "🎯 Training": page_training,
    "🔍 Predict": page_predict,
    "ℹ️ Model Info": page_model_info,
}

# Auth gate: regular users can only access Predict
if not st.session_state.get(AUTH_KEY, False) and nav != "🔍 Predict":
    st.session_state[NAV_KEY] = "🔍 Predict"
    nav = "🔍 Predict"

PAGES[nav]()
