import streamlit as st
import joblib
import os

# === Absolute paths to the files ===
vectorizer_path = r'c:\Users\ADMIN\.vscode\URa-Demo\vectorizer.joblib'
model_path = r'c:\Users\ADMIN\.vscode\URa-Demo\model.joblib'

# === Check if required files exist ===
missing_files = []
if not os.path.exists(vectorizer_path):
    missing_files.append(vectorizer_path)
if not os.path.exists(model_path):
    missing_files.append(model_path)

if missing_files:
    st.error(f"Required file(s) missing: {', '.join(missing_files)}")
else:
    try:
        # === Load the vectorizer and model ===
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)

        # === App UI ===
        st.title("📰 Fake News Detection")
        st.write("Enter the news article text below to check if it's likely real or fake:")

        # Optional styling for a larger text area
        st.markdown(
            """
            <style>
            .stTextArea textarea {
                height: 200px !important;
            }
            </style>
            """, unsafe_allow_html=True
        )

        user_input = st.text_area("News Article Text")

        if st.button("Check News"):
            if user_input.strip():
                # === Preprocess and Predict ===
                input_vector = vectorizer.transform([user_input])
                prediction = model.predict(input_vector)

                # === Display Result ===
                if prediction[0] == 1:
                    st.success("✅ The news article is likely to be **REAL**.")
                else:
                    st.error("⚠️ The news article is likely to be **FAKE**.")

                # Optional: Show prediction confidence if supported
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_vector)[0]
                    st.info(f"Confidence — Real: {proba[1]*100:.2f}%, Fake: {proba[0]*100:.2f}%")
            else:
                st.warning("Please enter some text to analyze.")
    
        st.markdown("---")
        st.info("Disclaimer: This tool is for demo purposes and may not be 100% accurate.")

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
        st.error("Please ensure the model and vectorizer files are correct and compatible.")