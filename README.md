# Next-word-and-sentence-predictor

# 🧠 LSTM Next Word Predictor

An interactive **Next Word Prediction and Text Generation** application built using **LSTM (Long Short-Term Memory)**, TensorFlow, Keras, and Streamlit.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nextwordpy-ep3zvape3v8ggp4bxdy3xg.streamlit.app/)
---

The application takes user-provided text and uses a trained LSTM model to predict the most probable next word. It can also generate multiple words automatically based on the user's input.

## 🚀 Features

* 🔮 Predict the next word from user-provided text
* 📊 Display multiple next-word predictions with probabilities
* 🌡️ Temperature control for prediction behavior
* ✨ Generate multiple words automatically
* ⚙️ Adjustable number of suggestions
* 🧠 Uses a trained LSTM neural network
* 🖥️ Interactive Streamlit web interface

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **LSTM**
* **NumPy**
* **Streamlit**
* **Pickle**

## 📂 Project Structure

```text
LSTM-Next-Word-Predictor/
│
├── next_word.py          # Streamlit application
├── lstm_model.h5         # Trained LSTM model
├── tokenizer.pkl         # Trained tokenizer
├── max_len.pkl           # Maximum sequence length
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## ⚙️ How It Works

The application follows these steps:

1. The user enters a piece of text.
2. The tokenizer converts the text into numerical sequences.
3. The sequence is padded to the required maximum length.
4. The trained LSTM model predicts probabilities for possible next words.
5. The application displays the highest-probability predictions.
6. For text generation, the predicted word is added to the input and the process is repeated.

## 🔮 Next Word Prediction

The application provides multiple prediction settings:

* **Number of suggestions:** Controls how many possible next words are displayed.
* **Temperature:** Controls the probability distribution of predictions.
* **Words to generate:** Controls how many words are generated automatically.

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/LSTM-Next-Word-Predictor.git
```

Navigate to the project directory:

```bash
cd LSTM-Next-Word-Predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run next_word.py
```

The application will open in your browser at the local Streamlit address.

## 🌐 Deployment

This project can be deployed using **Streamlit Community Cloud** by connecting the GitHub repository and selecting:

```text
Main file: next_word.py
```

## 📌 Example

Enter:

```text
The world as we have
```

The model will analyze the sequence and provide possible next-word predictions based on the trained LSTM model.

## 📚 Model

The application uses a trained LSTM model along with a tokenizer and maximum sequence length for preprocessing the input text.

The trained model is loaded from:

```text
lstm_model.h5
```

The tokenizer is loaded from:

```text
tokenizer.pkl
```

The maximum sequence length is loaded from:

```text
max_len.pkl
```

## 👨‍💻 Author

**Alok Gujar**

---

⭐ If you find this project useful, consider giving the repository a star!
