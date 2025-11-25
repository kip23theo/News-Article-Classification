import streamlit as st
import pickle
from PIL import Image
import io
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
LOGO_PATH = r"/mnt/data/cc134440-300f-4f2e-bbde-3f846c983806.png"  # <- uploaded image path (used as logo)
MODEL_PATH = "model.pkl"
VECT_PATH = "tfidf.pkl"

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Fake News Detector — NewsStyle",
    page_icon="🗞️",
    layout="wide"
)

# ---------- LOAD ASSETS ----------
@st.cache_resource
def load_model_and_vectorizer(mpath, vpath):
    model = pickle.load(open(mpath, "rb"))
    tfidf = pickle.load(open(vpath, "rb"))
    return model, tfidf

try:
    model, tfidf = load_model_and_vectorizer(MODEL_PATH, VECT_PATH)
except Exception as e:
    st.error("Error loading model or vectorizer. Please ensure model.pkl and tfidf.pkl are in the folder.")
    st.stop()

# ---------- STYLES (mature newspaper look) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Merriweather', Georgia, 'Times New Roman', serif;
        color: #111111;
        background-color: #FAFAFA;
    }
    .masthead {
        display:flex;
        align-items:center;
        gap:16px;
    }
    .title-large {
        font-size:42px;
        font-weight:700;
        margin:0;
        line-height:1;
    }
    .subtitle {
        font-size:14px;
        color:#444444;
        margin-top:6px;
    }
    .paper-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .result-true { color: #0b8a0b; font-weight:700; }
    .result-fake { color: #b00020; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER / MASTHEAD ----------
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((88, 88))
        st.image(logo, width=88)
    except Exception:
        st.markdown("🗞️")  # fallback icon

with col2:
    st.markdown('<div class="masthead">', unsafe_allow_html=True)
    st.markdown('<h1 class="title-large">Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A simple, reliable classifier in a mature news-style layout.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This tool classifies news text as **Real** or **Fake** using a TF-IDF + Logistic Regression model."
    )
    st.write("Model trained on a public dataset. Use results as guidance, not absolute truth.")
    st.markdown("---")
    st.header("Examples")
    if st.button("Load Real example"):
        st.session_state['input_text'] = (
            "The Indian Space Research Organisation (ISRO) successfully conducted a major test of its reusable launch vehicle "
            "on Saturday, marking a significant step toward reducing the cost of space missions. The test was carried out at "
            "the Aeronautical Test Range in Chitradurga, Karnataka."
        )
    if st.button("Load Fake example"):
        st.session_state['input_text'] = (
            "Scientists have confirmed that drinking cold water every morning can instantly cure all types of cancer within two weeks. "
            "The breakthrough was announced secretly on social media before being removed by government officials."
        )
    st.markdown("---")
    st.header("Controls")
    show_topk = st.slider("Show top important words", 0, 10, 5)
    st.markdown("Model and vectorizer are loaded from the local files `model.pkl` and `tfidf.pkl`.")
    st.markdown("---")
    st.caption("Made with ❤️ — Keep testing with real-world text.")

# ---------- MAIN INPUT / ANALYSIS ----------
st.markdown('<div class="paper-card">', unsafe_allow_html=True)
st.subheader("Paste the news article below")
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

user_input = st.text_area("", value=st.session_state['input_text'], height=240, key="text_area")

col_a, col_b = st.columns([1, 4])
with col_a:
    analyze = st.button("Analyze", use_container_width=True)
with col_b:
    st.write("")  # spacer
    st.write("")

if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please paste some text to analyze.")
    else:
        # transform and predict
        X_vect = tfidf.transform([user_input])
        # Predict label
        try:
            prob = model.predict_proba(X_vect)[0]
            confidence = float(np.max(prob))
        except Exception:
            # fallback if predict_proba not available
            pred = model.predict(X_vect)[0]
            confidence = None

        label = int(model.predict(X_vect)[0])
        if label == 1:
            st.markdown(f"<div class='result-true'>✔️ Classified as REAL</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-fake'>❌ Classified as FAKE</div>", unsafe_allow_html=True)

        if confidence is not None:
            st.write(f"**Confidence:** {confidence*100:.1f}%")
        else:
            st.write("**Confidence:** Not available for this model.")

        # Top words from input (explainability)
        if show_topk > 0:
            st.markdown("**Top words in this article (by TF-IDF score):**")
            try:
                # get tfidf features and scores for the single document
                feature_names = np.array(tfidf.get_feature_names_out())
                row = X_vect.toarray()[0]
                top_idx = row.argsort()[-show_topk:][::-1]
                top_words = [(feature_names[i], row[i]) for i in top_idx if row[i] > 0]
                if top_words:
                    df_top = pd.DataFrame(top_words, columns=["word", "tfidf_score"])
                    st.table(df_top)
                else:
                    st.write("No strong terms found (try a longer article).")
            except Exception:
                st.write("Unable to compute top words for this model configuration.")

        # Download result button
        result_text = f"Prediction: {'REAL' if label==1 else 'FAKE'}\n"
        if confidence is not None:
            result_text += f"Confidence: {confidence*100:.1f}%\n\n"
        result_text += "Article:\n" + user_input
        b = st.download_button(
            label="Download result as .txt",
            data=result_text,
            file_name="news_check_result.txt",
            mime="text/plain"
        )
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("**Notes:** This classifier is a simple model and may produce false positives/negatives. Use it as an aid, not a final authority.")
