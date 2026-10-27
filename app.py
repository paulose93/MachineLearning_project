import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detector", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

st.title("Fake News Detection Web App")
st.write("Analyze news articles to determine if they are likely Real or Fake.")

news_text = st.text_area("Paste the news article text here:", height=200)

if st.button("Analyze Article"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text with BERT..."):
            classifier = load_model()
            
            result = classifier(news_text, truncation=True, max_length=512)
            
            raw_label = result[0]['label']
            confidence = result[0]['score']
            
            # jy46604790 mapping: LABEL_0 is FAKE, LABEL_1 is REAL
            if raw_label in ["LABEL_0", "FAKE"]:
                display_label = "Fake News"
                is_fake = True
            else:
                display_label = "Real News"
                is_fake = False
            
            st.subheader("Analysis Result")
            if is_fake:
                st.error(f"**Classification:** {display_label}")
            else:
                st.success(f"**Classification:** {display_label}")
                
            st.write(f"**Confidence Score:** {confidence:.2%}")
            st.caption(f"Raw Model Output: `{raw_label}`")