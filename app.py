import streamlit as st
import torch
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from fsin_model import SIRALFractalNeuralNetwork

# Настройки страницы
st.set_page_config(page_title="Сфираль (SFIRAL): Фрактальная Нейронная Сеть", layout="wide")

# Заголовок
st.title("Сфираль (SFIRAL): Фрактальная Сфиральная Нейронная Сеть")
st.subheader("Интерфейс взаимодействия с моделью на русском языке.")

# Функция для визуализации графика
def plot_predictions(predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, marker='o', linestyle='-', label="Предсказания модели")
    plt.title("График предсказаний модели")
    plt.xlabel("Индекс")
    plt.ylabel("Значение предсказания")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Функция для выгрузки CSV
def create_csv(predictions):
    df = pd.DataFrame({"Predictions": predictions})
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

# Загрузка модели
st.sidebar.subheader("Загрузка обученной модели")
model_file = st.sidebar.file_uploader("Выберите файл модели (.pth)", type=["pth"])

if model_file:
    input_size = 10
    hidden_size = 32
    output_size = 1

    model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
    try:
        model.load_state_dict(torch.load(model_file))
        model.eval()
        st.sidebar.success("Модель успешно загружена!")
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки модели: {e}")
        st.stop()
else:
    st.sidebar.warning("Пожалуйста, загрузите файл модели.")
    st.stop()

# Ввод данных
st.subheader("Ввод данных")
num_inputs = st.number_input("Количество примеров для предсказания", min_value=1, max_value=100, value=10)
text_input = st.text_area("Введите данные для модели (через запятую для каждого примера):", "")
data = []

if text_input:
    try:
        data = [[float(x.strip()) for x in text_input.split(",")]]
    except ValueError:
        st.error("Ошибка: Проверьте формат данных. Ввод должен быть числовым.")
else:
    for i in range(num_inputs):
        st.write(f"Введите значения для примера {i + 1}:")
        example = []
        for j in range(input_size):
            value = st.number_input(f"Введите значение для параметра {j + 1}", value=0.5, step=0.1, key=f"input_{i}_{j}")
            example.append(value)
        data.append(example)

# Предсказания модели
if st.button("Выполнить предсказание"):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(data_tensor).numpy().flatten()
        st.success("Предсказания выполнены!")
        st.write("Предсказания модели:")
        st.table(predictions)

        # График предсказаний
        st.subheader("График предсказаний")
        buf = plot_predictions(predictions)
        st.image(buf)

        # Кнопка выгрузки
        csv = create_csv(predictions)
        st.download_button(
            label="Скачать предсказания в формате CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
