import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ======= Konfigurasi Streamlit =======
st.set_page_config(page_title="Color Picker Landowski", layout="centered")

# ======= CSS Styling =======
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    .title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }

    .color-box {
        width: 100px;
        height: 100px;
        margin: auto;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-in-out;
    }

    .color-box:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        cursor: pointer;
    }

    .hex-code {
        text-align: center;
        font-weight: bold;
        margin-top: 8px;
        font-family: monospace;
        animation: fadeIn 0.8s ease-in-out;
    }

    .main-block {
        background-color: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in-out;
    }
    </style>

    <div class="title">🎨 Color Picker Landowski</div>
""", unsafe_allow_html=True)

# ======= Kontrol UX =======
st.sidebar.header("⚙️ Pengaturan")

n_colors = st.sidebar.slider("🎨 Jumlah warna dominan", min_value=1, max_value=10, value=5)

# ======= Upload Gambar =======
uploaded_file = st.file_uploader("📂 Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ======= Fungsi Utama =======
def get_sorted_dominant_colors(image, n_colors):
    pixels = np.array(image).reshape((-1, 3))
    
    if n_colors == 1:
        # Hitung warna pixel terbanyak (mode)
        pixels_list = [tuple(pixel) for pixel in pixels]
        most_common_color = Counter(pixels_list).most_common(1)[0][0]
        return np.array([most_common_color])
    else:
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(pixels)
        counts = np.bincount(labels)
        sorted_indices = np.argsort(-counts)  # descending
        sorted_colors = kmeans.cluster_centers_[sorted_indices]
        return np.round(sorted_colors).astype(int)


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def calculate_brightness(rgb):
    return (rgb[0]*299 + rgb[1]*587 + rgb[2]*114) / 1000

# ======= Proses Utama =======
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    with st.spinner("🎯 Menganalisis warna dominan..."):
        colors = get_sorted_dominant_colors(image, n_colors)
        hex_colors = [rgb_to_hex(color) for color in colors]

    dominant_rgb = colors[0]
    dominant_hex = rgb_to_hex(dominant_rgb)
    brightness = calculate_brightness(dominant_rgb)
    text_color = '#ffffff' if brightness < 128 else '#000000'

    # CSS background & teks
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {dominant_hex};
            color: {text_color};
            transition: background-color 0.5s ease, color 0.5s ease;
        }}
        h1, h2, h3, h4, h5, h6,
        label, .css-1cpxqw2, .css-qrbaxs, .css-1kyxreq, .css-1n76uvr,
        .css-10trblm, .css-12oz5g7, .css-ffhzg2,
        .css-1j2dj06, button span {{
            color: {text_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Kontainer hasil
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

    st.subheader("🌈 Palet Warna Dominan")
    hex_html = ""
    for hex_color in hex_colors:
        hex_html += f"""
            <div style="display:inline-block; text-align:center; margin:10px;">
                <div class="color-box" style="background-color:{hex_color};"></div>
                <div class="hex-code" style="color:{text_color};">{hex_color}</div>
            </div>
        """
    st.markdown(hex_html, unsafe_allow_html=True)

    st.subheader("📊 Visualisasi Palet")
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color/255))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
