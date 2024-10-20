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
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return losses


# Функция для тестирования модели и расчета метрик
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    return accuracy, all_predictions, all_labels


# Функция для визуализации
def plot_loss(losses, model_info, hypothesis_number):
    plt.figure(figsize=(10, 5))
    plt.title(f"Training Loss - {model_info} (Hypothesis {hypothesis_number})")
    plt.plot(losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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

    losses = train_model(model, train_loader, criterion, optimizer, num_epochs)
    accuracy, predictions, labels = evaluate_model(model, test_loader)

    return accuracy, losses


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
    accuracy_single, losses_single = test_hypothesis(input_size, hidden_size, output_size, num_layers=1,
                                                     X_train=X_train, X_test=X_test,
                                                     y_train=y_train, y_test=y_test,
                                                     num_epochs=num_epochs, batch_size=batch_size)
    print(f'Точность однослойной LSTM: {accuracy_single:.2f}%')

    accuracy_double, losses_double = test_hypothesis(input_size, hidden_size, output_size, num_layers=2,
                                                     X_train=X_train, X_test=X_test,
                                                     y_train=y_train, y_test=y_test,
                                                     num_epochs=num_epochs, batch_size=batch_size)
    print(f'Точность двухслойной LSTM: {accuracy_double:.2f}%')

    plot_loss(losses_single, "Однослойная LSTM", 1)
    plot_loss(losses_double, "Двухслойная LSTM", 1)

    print("Гипотеза 2: Увеличение hidden_size с 128 до 256")
    accuracy_small_hidden, losses_small_hidden  = test_hypothesis(input_size, 128, output_size, num_layers=1,
                                               X_train=X_train, X_test=X_test,
                                               y_train=y_train, y_test=y_test,
                                               num_epochs=num_epochs, batch_size=batch_size)
    accuracy_large_hidden, losses_large_hidden  = test_hypothesis(input_size, 256, output_size, num_layers=1,
                                               X_train=X_train, X_test=X_test,
                                               y_train=y_train, y_test=y_test,
                                               num_epochs=num_epochs, batch_size=batch_size)
    print(f'Точность модели с hidden_size=128: {accuracy_small_hidden:.2f}%')
    print(f'Точность модели с hidden_size=256: {accuracy_large_hidden:.2f}%')

    plot_loss(losses_small_hidden, "LSTM hidden_size=128", 2)
    plot_loss(losses_large_hidden, "LSTM hidden_size=256", 2)

    print("Гипотеза 3: Увеличение числа эпох с 10 до 20")
    accuracy_short_epochs, losses_short_epochs  = test_hypothesis(input_size, hidden_size, output_size, num_layers=1,
                                               X_train=X_train, X_test=X_test,
                                               y_train=y_train, y_test=y_test,
                                               num_epochs=10, batch_size=batch_size)
    accuracy_long_epochs, losses_long_epochs  = test_hypothesis(input_size, hidden_size, output_size, num_layers=1,
                                              X_train=X_train, X_test=X_test,
                                              y_train=y_train, y_test=y_test,
                                              num_epochs=20, batch_size=batch_size)
    print(f'Точность модели с 10 эпохами: {accuracy_short_epochs:.2f}%')
    print(f'Точность модели с 20 эпохами: {accuracy_long_epochs:.2f}%')

    plot_loss(losses_short_epochs, "10 epochs", 3)
    plot_loss(losses_long_epochs, "20 epochs", 3)