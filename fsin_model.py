import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Генерация синтетических данных
def generate_data(num_samples, input_size):
    x = torch.rand(num_samples, input_size)
    y = torch.sum(x, dim=1, keepdim=True)
    return x, y

# Нормализация данных
def normalize_data(x):
    return (x - x.mean(dim=0)) / x.std(dim=0)

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
        x = self.output(x)
        return x

# Обучение модели
def train_model(model, x_train, y_train, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return losses

# Визуализация потерь
def plot_losses(losses):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# Основная функция запуска обучения
def run_training():
    print("Запуск обучения модели...")
    
    # Параметры
    input_size = 10
    hidden_size = 32
    output_size = 1
    num_samples = 500

    # Генерация и нормализация данных
    x_train, y_train = generate_data(num_samples, input_size)
    x_train = normalize_data(x_train)
    
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    # Инициализация модели
    model = SIRALFractalNeuralNetwork(input_size, hidden_size, output_size)
    print(model)

    # Обучение модели
    losses = train_model(model, x_train, y_train)

    # Сохранение модели
    torch.save(model.state_dict(), "fsin_model.pth")
    print("Модель успешно сохранена как fsin_model.pth")

    # Построение графика потерь
    plot_losses(losses)
