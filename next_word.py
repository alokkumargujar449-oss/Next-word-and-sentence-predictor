import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

@st.cache_resource
def load_assets():

    # Load trained LSTM model
    model = load_model(
        "lstm_model (1).h5",
        compile=False
    )

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    # Load maximum sequence length
    with open("max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_assets()


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f0f2f6;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-word {
        font-size: 32px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 LSTM Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the next word using a trained LSTM neural network'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of suggestions",
    min_value=1,
    max_value=10,
    value=5
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

num_words = st.sidebar.slider(
    "Words to generate",
    min_value=1,
    max_value=20,
    value=5
)

st.sidebar.divider()

st.sidebar.write(
    f"**Vocabulary size:** {len(tokenizer.word_index)}"
)

st.sidebar.write(
    f"**Maximum sequence length:** {max_len}"
)


# ============================================================
# FUNCTION TO PREPARE INPUT
# ============================================================

def prepare_input(text):

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return None

    # Pad sequence to the model's required length
    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    return padded_sequence


# ============================================================
# FUNCTION TO PREDICT NEXT WORD
# ============================================================

def predict_next_words(text, top_k=5, temperature=1.0):

    input_sequence = prepare_input(text)

    if input_sequence is None:
        return []

    # Model prediction
    prediction = model.predict(
        input_sequence,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Temperature scaling
    # --------------------------------------------------------

    prediction = np.asarray(prediction, dtype=np.float64)

    prediction = np.log(
        np.maximum(prediction, 1e-12)
    ) / temperature

    prediction = np.exp(
        prediction - np.max(prediction)
    )

    prediction = prediction / np.sum(prediction)

    # --------------------------------------------------------
    # Get highest probability word IDs
    # --------------------------------------------------------

    word_indices = np.argsort(
        prediction
    )[::-1]

    results = []

    for index in word_indices:

        # Convert index to word
        word = tokenizer.index_word.get(
            int(index)
        )

        if word is None:
            continue

        probability = prediction[index]

        results.append(
            (word, probability)
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# FUNCTION TO GENERATE MULTIPLE WORDS
# ============================================================

def generate_text(text, number_of_words, temperature=1.0):

    generated_text = text.strip()

    for _ in range(number_of_words):

        predictions = predict_next_words(
            generated_text,
            top_k=1,
            temperature=temperature
        )

        if not predictions:
            break

        next_word = predictions[0][0]

        generated_text += " " + next_word

    return generated_text


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader("✍️ Enter your text")

text = st.text_area(
    "Start typing:",
    height=150,
    placeholder=(
        "Example:\n"
        "The world as we have created it"
    )
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    predict_button = st.button(
        "🔮 Predict Next Word",
        use_container_width=True
    )

with col2:

    generate_button = st.button(
        "✨ Generate Text",
        use_container_width=True
    )


# ============================================================
# NEXT WORD PREDICTION
# ============================================================

if predict_button:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text first."
        )

    else:

        predictions = predict_next_words(
            text,
            top_k=top_k,
            temperature=temperature
        )

        if not predictions:

            st.error(
                "Could not predict a word. "
                "Try entering different text."
            )

        else:

            # Best prediction
            best_word = predictions[0][0]

            st.markdown(
                """
                <div class="prediction-box">

                <div>Predicted Next Word</div>

                <div class="prediction-word">
                %s
                </div>

                </div>
                """
                % best_word,
                unsafe_allow_html=True
            )

            st.subheader("📊 Other Predictions")

            for rank, (word, probability) in enumerate(
                predictions,
                start=1
            ):

                percentage = probability * 100

                st.write(
                    f"**{rank}. {word}** — "
                    f"{percentage:.2f}%"
                )

                st.progress(
                    float(probability)
                )


# ============================================================
# GENERATE TEXT
# ============================================================

if generate_button:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text first."
        )

    else:

        with st.spinner(
            "🧠 Generating text..."
        ):

            generated_text = generate_text(
                text,
                num_words,
                temperature
            )

        st.subheader("✨ Generated Text")

        st.info(
            generated_text
        )

        st.text_area(
            "📋 Copy generated text",
            value=generated_text,
            height=120
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built using TensorFlow, Keras, LSTM and Streamlit"
)