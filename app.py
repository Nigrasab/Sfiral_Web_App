import os
import streamlit as st
import torch
from torch import nn
import requests

# Определение модели
class SIRALFractalNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SIRALFractalNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

# Загрузка модели из репозитория
def download_model(repo_url, model_filename):
    local_filename = model_filename
    if not os.path.exists(local_filename):
        st.info("Загрузка модели из репозитория...")
        response = requests.get(f"{repo_url}/raw/main/{model_filename}", stream=True)
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        st.success("Модель успешно загружена!")
    return local_filename

# Основной интерфейс приложения
def main():
    st.title("Сфираль (SFIRAL): Фрактальная Сфиральная Нейронная Сеть")
    st.markdown("Интерфейс взаимодействия с моделью на русском языке.")

    # Параметры модели
    input_size = 10
    hidden_size = 32
    output_size = 1

    # Загрузка модели
    repo_url = "https://github.com/Nigrasab/Sfiral_Web_App"
    model_filename = "fsin_model.pth"
    model_path = download_model(repo_url, model_filename)

    # Загрузка модели в память
    model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Ввод данных пользователем
    st.header("Ввод данных")
    user_input = []
    for i in range(input_size):
        value = st.number_input(f"Введите значение для параметра {i + 1}:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        user_input.append(value)

    # Прогнозирование
    if st.button("Запустить модель"):
        input_tensor = torch.tensor([user_input])
        prediction = model(input_tensor).item()
        st.success(f"Результат предсказания: {prediction:.4f}")

# Запуск приложения
if __name__ == "__main__":
    main()
