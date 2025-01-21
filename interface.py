import streamlit as st
import requests

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/analyze"

# Helper function to display sentiment with colors
def get_sentiment_color(label):
    if label == "POSITIVE":
        return "green"
    elif label == "NEGATIVE":
        return "red"
    else:
        return "blue"

# Streamlit app
def main():
    # Page configuration
    st.set_page_config(
        page_title="Sentiment & Category Analysis",
        page_icon="🤖",
        layout="wide",
    )

    # Header Section
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 36px;
                font-weight: bold;
                color: #FFFFFF;
                text-align: center;
                margin-bottom: 10px;
            }
            .sub-header {
                font-size: 16px;
                color: #555;
                text-align: center;
                margin-bottom: 30px;
            }
        </style>
        <div>
            <h1 class="main-header">🤖 Sentiment & Category Analysis</h1>
            <p class="sub-header">
                Analyze text for sentiment and category classification with confidence scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        # Input Section
        st.markdown("### 📥 Input Text")
        text = st.text_area(
            "Enter the text you want to analyze",
            placeholder="Type your text here...",
            height=200,
        )

        if st.button("🔍 Analyze", key="analyze_button"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        # Send a POST request to the Flask API
                        response = requests.post(API_URL, json={"text": text})

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state["result"] = result  # Save result in session state
                        else:
                            st.error(f"❌ Error: {response.json().get('error', 'Unknown error occurred')}")
                    except Exception as e:
                        st.error(f"⚠️ An error occurred: {e}")
            else:
                st.warning("⚠️ Please enter some text before analyzing!")

    with col2:
        # Output Section
        st.markdown("### 📤 Analysis Results")
        if "result" in st.session_state:
            result = st.session_state["result"]

            # Sentiment Analysis Result
            sentiment_output = result.get("sentiment", [])
            if sentiment_output:
                sentiment_label = sentiment_output[0]["label"]
                sentiment_score = sentiment_output[0]["score"]
                sentiment_color = get_sentiment_color(sentiment_label)

                st.markdown(
                    f"""
                    <div style="background-color: #f9f9f9; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                        <h4 style="color: {sentiment_color}; text-align: center;">
                            Sentiment: {sentiment_label}
                        </h4>
                        <p style="text-align: center;">
                            Confidence Score: <b>{sentiment_score:.4f}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Zero-Shot Classification Result
            category = result.get("classification", "N/A")
            confidence = result.get("confidence", 0)

            st.markdown(
                f"""
                <div style=" border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <h4 style="color: #007acc; text-align: center;">
                        Category: {category}
                    </h4>
                    <p style="text-align: center;">
                        Confidence Score: <b>{confidence:.2f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Enter text and click 'Analyze' to see the results!")

    # Footer Section
    st.markdown(
        """
        ---
        <div style="text-align: center; color: #aaa; font-size: 14px;">
            Built with ❤️ using Streamlit | <a href="https://github.com/" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
