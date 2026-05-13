import torch
import torch.nn as nn
import torch.optim as optim
from model import DefectClassifier
from data_loader import get_data_loaders
import os


def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device used for training: {device}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, val_loader, classes = get_data_loaders(base_dir, batch_size=32)

    model = DefectClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    print("Training starts...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)

                _, predicted_classes = torch.max(val_outputs, dim=1)

                total_samples += val_labels.size(0)
                correct_predictions += (predicted_classes == val_labels).sum().item()

        accuracy = (correct_predictions / total_samples) * 100
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        save_path = os.path.join(base_dir, '..', 'models', 'defect_model_weights.pth')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved successfully to {save_path}")

if __name__ == "__main__":
    train_model()