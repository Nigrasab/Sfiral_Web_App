import streamlit as st
import torch
from fsin_model import SIRALFractalNeuralNetwork

# Параметры модели
input_size = 10
hidden_size = 32
output_size = 1
model_path = "https://raw.githubusercontent.com/YourUsername/YourRepoName/main/fsin_model.pth"  # Укажите правильный URL

# Функция для загрузки модели из репозитория
@st.cache_resource
def load_model():
    try:
        # Загрузка модели
        model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Интерфейс Streamlit
st.title("Sfiral Prediction Interface")
st.write("Use this interface to interact with the Sfiral Model.")

# Загрузка модели
model = load_model()
if model is None:
    st.error("Model could not be loaded. Please check the repository URL.")
else:
    st.success("Model successfully loaded!")

    # Ввод пользовательских данных
    st.subheader("Enter Data for Prediction")
    user_input = []
    for i in range(input_size):
        value = st.number_input(f"Input {i + 1}", value=0.0, step=0.1)
        user_input.append(value)

    if st.button("Make Prediction"):
        input_tensor = torch.tensor([user_input], dtype=torch.float32)
        prediction = model(input_tensor).item()
        st.write(f"Prediction: {prediction}")
