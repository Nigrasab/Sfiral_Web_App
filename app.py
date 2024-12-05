import streamlit as st
import torch
from fsin_model import SIRALFractalNeuralNetwork
import requests
import io
import pandas as pd
import matplotlib.pyplot as plt

# Функция загрузки модели из репозитория
@st.cache_resource
def load_model_from_repo(repo_url):
    try:
        response = requests.get(repo_url)
        response.raise_for_status()  # Проверка успешности запроса
        model_state = torch.load(io.BytesIO(response.content))
        model = SIRALFractalNeuralNetwork(input_size=10, hidden_size=32, output_size=1)
        model.load_state_dict(model_state)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load the model from repository: {e}")
        return None

# Основной код приложения
def main():
    st.title("SIRAL Model Interface")

    # Укажите URL репозитория для загрузки модели
    repo_url = st.text_input("Enter Model Repository URL:", 
                             "https://example.com/path/to/fsin_model.pth")

    if st.button("Load Model"):
        model = load_model_from_repo(repo_url)
        if model:
            st.success("Model successfully loaded!")
        else:
            st.error("Model could not be loaded.")

    # Проверка, если модель загружена
    if 'model' in locals() and model:
        # Опции для пользовательских данных
        st.subheader("Generate Predictions")
        user_input = st.text_area(
            "Enter your data as comma-separated values (10 values per row):",
            "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
        )

        # Преобразование пользовательских данных
        try:
            data = [list(map(float, row.split(','))) for row in user_input.split('\n') if row]
            input_tensor = torch.tensor(data)
            if input_tensor.shape[1] != 10:
                raise ValueError("Each row must have exactly 10 values.")
        except Exception as e:
            st.error(f"Invalid input data: {e}")
            return

        # Генерация предсказаний
        predictions = model(input_tensor)
        st.write("Predictions:", predictions.detach().numpy())

        # Визуализация предсказаний
        st.subheader("Prediction Graph")
        df = pd.DataFrame(predictions.detach().numpy(), columns=["Prediction"])
        st.line_chart(df)

# Запуск приложения
if __name__ == "__main__":
    main()
