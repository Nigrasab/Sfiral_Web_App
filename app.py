import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fsin_model import SIRALFractalNeuralNetwork  # Убедитесь, что модель находится в той же папке

# Функция для визуализации предсказаний
def plot_predictions(predictions):
    fig, ax = plt.subplots()
    ax.plot(predictions, marker="o", label="Predictions")
    ax.set_title("Model Predictions")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Prediction Value")
    ax.legend()
    st.pyplot(fig)

# Заголовок интерфейса
st.title("Интерфейс модели SFIRAL")

# Загрузка модели
st.header("Загрузка обученной модели")
uploaded_model = st.file_uploader("Выберите файл модели (.pth)", type="pth")

# Проверка, загрузил ли пользователь модель
model = None
if uploaded_model is not None:
    try:
        input_size = 10
        hidden_size = 32
        output_size = 1
        model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(uploaded_model))
        model.eval()
        st.success("Модель успешно загружена!")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")

# Блок предсказаний
if model is not None:
    st.header("Предсказания")
    
    # Выбор способа ввода данных
    input_method = st.radio("Выберите способ ввода данных:", ["Сгенерировать случайные данные", "Ввести данные вручную"])
    
    if input_method == "Сгенерировать случайные данные":
        num_samples = st.number_input("Количество примеров для предсказания:", min_value=1, max_value=100, value=10)
        if st.button("Выполнить предсказание"):
            new_data = torch.rand(int(num_samples), input_size)
            predictions = model(new_data).detach().numpy()
            st.write("Предсказания модели:", predictions)
            plot_predictions(predictions)
    
    elif input_method == "Ввести данные вручную":
        user_input = st.text_area("Введите данные в формате CSV (каждая строка - новый пример):", "0.5, 0.3, 0.8, 0.1, 0.2, 0.7, 0.9, 0.6, 0.4, 0.5")
        if st.button("Выполнить предсказание"):
            try:
                # Преобразование пользовательских данных
                data = np.array([list(map(float, row.split(','))) for row in user_input.splitlines()])
                if data.shape[1] != input_size:
                    st.error(f"Ошибка: Должно быть {input_size} признаков в каждом примере.")
                else:
                    new_data = torch.tensor(data, dtype=torch.float32)
                    predictions = model(new_data).detach().numpy()
                    st.write("Предсказания модели:", predictions)
                    plot_predictions(predictions)
            except Exception as e:
                st.error(f"Ошибка обработки данных: {e}")
