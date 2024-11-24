import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings


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
    plt.title(f" {model_info}")
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


def dynamic_quantize_model(model):
    model.eval()
    model = torch.quantization.prepare(model, inplace=False)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )
    return quantized_model


def compare_models(original_model, quantized_model, test_loader):
    start_time = time.time()
    original_loss, original_accuracy = evaluate_model(original_model, test_loader, nn.CrossEntropyLoss())
    original_inference_time = time.time() - start_time

    start_time = time.time()
    quantized_loss, quantized_accuracy = evaluate_model(quantized_model, test_loader, nn.CrossEntropyLoss())
    quantized_inference_time = time.time() - start_time

    original_size = sum(p.numel() for p in original_model.parameters()) * 4 / (1024 ** 2)  # в MB
    quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / (1024 ** 2)  # в MB

    print(f"Исходная модель: Время инференса = {original_inference_time:.4f} сек, Размер = {original_size:.2f} MB")
    print(f"Квантованная модель: Время инференса = {quantized_inference_time:.4f} сек, Размер = {quantized_size:.2f} MB")

    return original_accuracy, quantized_accuracy



def show_incorrect_predictions(model, test_loader, device):
    model.eval()
    incorrect_samples = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            incorrect = predicted != labels
            for i in range(len(labels)):
                if incorrect[i]:
                    incorrect_samples.append((inputs[i], labels[i], predicted[i]))

            if len(incorrect_samples) >= 5:
                break

    for idx, (input_data, true_label, pred_label) in enumerate(incorrect_samples):

        text = ' '.join([str(x) for x in input_data.cpu().numpy()])

        print(f"Sample {idx}:")
        print(f"Текст: {text}")
        print(f"Истинное: {true_label.item()}, Предсказано: {pred_label.item()}")
        print("=" * 50)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Исходная модель")
    train_losses_single, test_losses_single, train_acc_single, test_acc_single = test_hypothesis(input_size,
                                                                                                 hidden_size,
                                                                                                 output_size,
                                                                                                 num_layers=1,
                                                                                                 X_train=X_train,
                                                                                                 X_test=X_test,
                                                                                                 y_train=y_train,
                                                                                                 y_test=y_test,
                                                                                                 num_epochs=num_epochs,
                                                                                                 batch_size=batch_size)

    plot_loss(train_losses_single, "Потери для тренировочной выборки", 1, "loss")
    plot_loss(test_losses_single, "Потери для тестовой выборки", 1, "loss")
    plot_loss(train_acc_single, "Точность для тренировочной выборки", 1, "accuracy")
    plot_loss(test_acc_single, "Точность для тестовой выборки", 1, "accuracy")

    original_model = create_model(input_size, hidden_size, output_size)
    quantized_model = dynamic_quantize_model(original_model)

    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = prepare_data(X_train, X_test, y_train, y_test, batch_size)

    print("Квантованная модель")
    train_losses_quantized, test_losses_quantized, train_acc_quantized, test_acc_quantized = test_hypothesis(input_size,
                                                                                                             hidden_size,
                                                                                                             output_size,
                                                                                                             num_layers=1,
                                                                                                             X_train=X_train,
                                                                                                             X_test=X_test,
                                                                                                             y_train=y_train,
                                                                                                             y_test=y_test,
                                                                                                             num_epochs=num_epochs,
                                                                                                             batch_size=batch_size)

    #Графики для квантованной модели
    plot_loss(train_losses_quantized, "Потери для тренировочной выборки (Квантованная)", 1, "loss")
    plot_loss(test_losses_quantized, "Потери для тестовой выборки (Квантованная)", 1, "loss")
    plot_loss(train_acc_quantized, "Точность для тренировочной выборки (Квантованная)", 1, "accuracy")
    plot_loss(test_acc_quantized, "Точность для тестовой выборки (Квантованная)", 1, "accuracy")

    compare_models(original_model, quantized_model, test_loader)

    print("Ошибочные предсказания для оригинальной модели:")
    show_incorrect_predictions(original_model, test_loader, device)

    print("Ошибочные предсказания для квантованной модели:")
    show_incorrect_predictions(quantized_model, test_loader, device)
