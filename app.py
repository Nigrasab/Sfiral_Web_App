import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from fsin_model import SIRALFractalNeuralNetwork  # Убедитесь, что модель доступна в той же папке

# Функция для построения графика
def plot_predictions(predictions):
    fig, ax = plt.subplots()
    ax.plot(range(len(predictions)), predictions, marker='o', linestyle='-')
    ax.set_title("Предсказания модели")
    ax.set_xlabel("Номер примера")
    ax.set_ylabel("Значение")
    return fig

# Загрузка модели
def load_model():
    input_size = 10
    hidden_size = 32
    output_size = 1

    model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
    model_path = "fsin_model.pth"  # Убедитесь, что этот файл находится в репозитории
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Интерфейс Streamlit
st.title("Сфираль (SFIRAL): Фрактальная Сфиральная Нейронная Сеть")
st.write("Интерфейс взаимодействия с моделью на русском языке.")

# Раздел загрузки модели
st.header("Загрузка обученной модели")
try:
    model = load_model()
    st.success("Модель успешно загружена!")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {str(e)}")

# Ввод данных
st.header("Ввод данных")
st.write("Введите числовые значения для параметров.")

inputs = []
for i in range(10):
    value = st.number_input(f"Введите значение для параметра {i + 1}:", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    inputs.append(value)

inputs_tensor = torch.tensor([inputs])

# Ввод текста
st.header("Текстовый ввод")
text_input = st.text_input("Введите текстовую информацию:")
st.write(f"Вы ввели: {text_input}")

# Предсказания модели
if st.button("Выполнить предсказание"):
    if model:
        predictions = model(inputs_tensor).detach().numpy().flatten()
        st.success("Предсказания выполнены успешно!")
        st.write("Предсказания модели:", predictions)

        # Построение графика
        st.subheader("График предсказаний")
        fig = plot_predictions(predictions)
        st.pyplot(fig)

        # Сохранение предсказаний
        st.subheader("Выгрузка предсказаний")
        df = pd.DataFrame({"Predictions": predictions})
        csv = df.to_csv(index=False)
        st.download_button(
            label="Скачать результаты в формате CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.error("Модель не загружена. Пожалуйста, загрузите модель и попробуйте снова.")
