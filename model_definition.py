import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class FakeNewsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(FakeNewsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Функция для инициализации модели
def create_model(input_size, hidden_size, output_size, num_layers=1):
    model = FakeNewsLSTM(input_size, hidden_size, output_size, num_layers)
    return model


# Функция для подготовки данных
def prepare_data(X_train, X_test, y_train, y_test, batch_size=64):
    X_train = X_train.clone().detach().float()
    X_test = X_test.clone().detach().float()

    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    y_train = y_train.clone().detach().long()
    y_test = y_test.clone().detach().long()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Функция для обучения модели
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training step
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Evaluate on test set at each epoch
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return train_losses, test_losses, train_accuracies, test_accuracies


# Функция для тестирования модели и расчета метрик
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# Функция для визуализации
def plot_loss(losses, model_info, hypothesis_number, plot_type):
    plt.figure(figsize=(10, 5))
    plt.title(f" {model_info} (Hypothesis {hypothesis_number})")
    plt.plot(losses, label=plot_type)
    plt.xlabel("Epoch")
    plt.ylabel(plot_type)
    plt.legend()
    plt.show()


# Функция для визуализации предсказаний
def show_predictions(predictions, labels, num_samples=2):
    print("Примеры корректных предсказаний и ошибок:")
    correct = [i for i in range(len(predictions)) if predictions[i] == labels[i]]
    incorrect = [i for i in range(len(predictions)) if predictions[i] != labels[i]]

    print("\nКорректные предсказания:")
    for idx in correct[:num_samples]:
        print(f"Sample {idx}: Предсказано {predictions[idx]}, Истинное {labels[idx]}")

    print("\nОшибочные предсказания:")
    for idx in incorrect[:num_samples]:
        print(f"Sample {idx}: Предсказано {predictions[idx]}, Истинное {labels[idx]}")


# Функция для обработки данных
def preprocess_data(texts, labels):
    vectorizer = CountVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
    y = [0 if label == 'FAKE' else 1 for label in labels]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Функция для проверки гипотез
def test_hypothesis(input_size, hidden_size, output_size, num_layers, X_train, X_test, y_train, y_test, num_epochs,
                    batch_size):
    model = create_model(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, test_loader = prepare_data(X_train, X_test, y_train, y_test, batch_size)

    train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader,
                                                                               criterion, optimizer, num_epochs)

    return train_losses, test_losses, train_accuracies, test_accuracies


# Пример использования модели
if __name__ == "__main__":
    input_size = 100
    hidden_size = 128
    output_size = 2
    num_epochs = 10
    batch_size = 64

    data = pd.read_csv('news.csv')

    news_texts = data['text'].tolist()
    labels = data['label'].tolist()

    X, y = preprocess_data(news_texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Гипотеза 1: Однослойная LSTM vs Двухслойная LSTM")
    print("Однослойная")
    train_losses_single, test_losses_single, train_acc_single, test_acc_single = test_hypothesis(input_size, hidden_size, output_size, num_layers=1,
                                                                                                  X_train=X_train, X_test=X_test,
                                                                                                  y_train=y_train, y_test=y_test,
                                                                                                  num_epochs=num_epochs, batch_size=batch_size)
    print("Двухслойная")
    train_losses_double, test_losses_double, train_acc_double, test_acc_double = test_hypothesis(input_size, hidden_size, output_size, num_layers=2,
                                                                                                  X_train=X_train, X_test=X_test,
                                                                                                  y_train=y_train, y_test=y_test,
                                                                                                  num_epochs=num_epochs, batch_size=batch_size)

    plot_loss(train_losses_single, "Потери для тренировочной выборки однослойной LSTM ", 1,"loss")
    plot_loss(train_losses_double, "Потери для тренировочной выборки двухслойной LSTM", 1,"loss")
    plot_loss(test_losses_single, "Потери для тестовой выборки однослойной LSTM ", 1,"loss")
    plot_loss(test_losses_double, "Потери для тестовой выборки двухслойной LSTM", 1, "loss")
    plot_loss(train_acc_single, "Точность для тренировочной выборки однослойной LSTM ", 1, "accuracy")
    plot_loss(train_acc_double, "Точность для тренировочной выборки двухслойной LSTM", 1, "accuracy")
    plot_loss(test_acc_single, "Точность для тестовой выборки однослойной LSTM ", 1, "accuracy")
    plot_loss(test_acc_double, "Точность для тестовой выборки двухслойной LSTM", 1, "accuracy")

    print("Гипотеза 2: Увеличение hidden_size с 128 до 256")
    print("128")
    train_losses_small_hidden, test_losses_small_hidden, train_acc_small_hidden, test_acc_small_hidden = test_hypothesis(
        input_size, 128, output_size, num_layers=1,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        num_epochs=num_epochs, batch_size=batch_size)
    print("256")
    train_losses_big_hidden, test_losses_big_hidden, train_acc_big_hidden, test_acc_big_hidden = test_hypothesis(
        input_size, 256, output_size, num_layers=1,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        num_epochs=num_epochs, batch_size=batch_size)
    plot_loss(train_losses_small_hidden, "Потери для тренировочной выборки hidden_size=128 ", 2,"loss")
    plot_loss(train_losses_big_hidden, "Потери для тренировочной выборки hidden_size=256", 2,"loss")
    plot_loss(test_losses_small_hidden, "Потери для тестовой выборки hidden_size=128 ", 2,"loss")
    plot_loss(test_losses_big_hidden, "Потери для тестовой выборки hidden_size=256", 2,"loss")
    plot_loss(train_acc_small_hidden, "Точность для тренировочной выборки hidden_size=128 ", 2, "accuracy")
    plot_loss(train_acc_big_hidden, "Точность для тренировочной выборки hidden_size=256", 2, "accuracy")
    plot_loss(test_acc_small_hidden, "Точность для тестовой выборки hidden_size=128 ", 2, "accuracy")
    plot_loss(test_acc_big_hidden, "Точность для тестовой выборки hidden_size=256", 2, "accuracy")

    print("Гипотеза 3: Увеличение числа эпох с 10 до 20")
    print("10 эпох")
    train_losses_short_epochs, test_losses_short_epochs, train_acc_short_epochs, test_acc_short_epochs = test_hypothesis(
        input_size, hidden_size, output_size, num_layers=1,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        num_epochs=10, batch_size=batch_size)
    print("20 эпох")
    train_losses_long_epochs, test_losses_long_epochs, train_acc_long_epochs, test_acc_long_epochs = test_hypothesis(
        input_size, hidden_size, output_size, num_layers=1,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        num_epochs=20, batch_size=batch_size)

    plot_loss(train_losses_short_epochs, "Потери для тренировочной выборки 10 эпох ", 3, "loss")
    plot_loss(train_losses_long_epochs, "Потери для тренировочной выборки 20 эпох", 3, "loss")
    plot_loss(test_losses_short_epochs, "Потери для тестовой выборки 10 эпох ", 3, "loss")
    plot_loss(test_losses_long_epochs, "Потери для тестовой выборки 20 эпох", 3, "loss")
    plot_loss(train_acc_short_epochs, "Точность для тренировочной выборки 10 эпох ", 3, "accuracy")
    plot_loss(train_acc_long_epochs, "Точность для тренировочной выборки 20 эпох", 3, "accuracy")
    plot_loss(test_acc_short_epochs, "Точность для тестовой выборки 10 эпох", 3, "accuracy")
    plot_loss(test_acc_long_epochs, "Точность для тестовой выборки 20 эпох", 3, "accuracy")