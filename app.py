import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- Cấu hình giao diện ---
st.set_page_config(page_title="Tóm Tắt Văn Bản", page_icon="📝", layout="centered")

# --- CSS LED RGB động + hiệu ứng Liquid Glass ---
st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        margin: 0;
        padding: 0;
        height: 100%;
        background: linear-gradient(270deg, #ff4b4b, #ffcc00, #00ffcc, #0099ff, #cc00ff);
        background-size: 1000% 1000%;
        animation: rgbBackground 20s ease infinite;
    }

    @keyframes rgbBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        padding: 3rem;
        margin-top: 4rem;
    }

    .stTextArea textarea {
        font-size: 16px !important;
        padding: 15px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.15);
        color: #000;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.4);
    }

    .stButton button:hover {
        background-color: #e03a3a;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
    }

    .title-style {
        color: #ffffff;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 0px 0px 10px #000000;
    }

    .summary-style {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        font-size: 18px;
        color: #000;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load mô hình T5 ---
@st.cache_resource
def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    return tokenizer, model

tokenizer, model = load_model_tokenizer()

# --- Hàm tóm tắt ---
def summarize_text_vi(text):
    input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids
    summary_ids = model.generate(
        input_ids,
        max_length=100,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

# --- Giao diện chính ---
st.markdown("<h1 class='title-style'>📝 Ứng Dụng Tóm Tắt Văn Bản</h1>", unsafe_allow_html=True)
st.markdown("### 📄 Nhập nội dung bên dưới để tiến hành tóm tắt:")

input_text = st.text_area("✏️ Nội dung văn bản:", height=200, placeholder="Nhập văn bản tại đây...")

if st.button("🚀 Tóm tắt ngay!"):
    if input_text.strip():
        with st.spinner("⏳ Đang tóm tắt, vui lòng đợi..."):
            progress_bar = st.progress(0)
            for percent_complete in range(0, 101, 10):
                time.sleep(0.03)
                progress_bar.progress(percent_complete)

            try:
                summary = summarize_text_vi(input_text)
                st.success("✅ Kết quả tóm tắt:")
                st.markdown(f"<div class='summary-style'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Lỗi khi tóm tắt: {str(e)}")
    else:
        st.warning("⚠️ Vui lòng nhập văn bản trước khi nhấn nút.")
