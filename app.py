import os
import requests
import torch
import streamlit as st
from fsin_model import SIRALFractalNeuralNetwork

# Загрузка модели из GitHub
MODEL_URL = "https://github.com/Nigrasab/Sfiral_Web_App/raw/main/fsin_model.pth"
MODEL_PATH = "fsin_model.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")

# Загрузка модели
def load_model():
    model = SIRALFractalNeuralNetwork(input_size=10, hidden_size=32, output_size=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the download process.")
        return None

# Отображение интерфейса
def main():
    st.title("Sfiral Model Interface")
    st.write("Input your data and see predictions.")

    # Загрузка модели
    download_model()
    model = load_model()
    if model is None:
        return

    # Таблица ввода данных
    st.subheader("Input Data")
    data = []
    for i in range(10):
        data.append(st.number_input(f"Input feature {i + 1}", min_value=0.0, max_value=10.0, value=0.0))

    # Кнопка предсказания
    if st.button("Predict"):
        input_tensor = torch.tensor([data], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)
        st.success(f"Prediction: {prediction.item():.4f}")

if __name__ == "__main__":
    main()
