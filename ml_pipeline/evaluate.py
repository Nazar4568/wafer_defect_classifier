from src.model import DefectClassifier
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_data_loaders
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device used: {device}")

    model = DefectClassifier().to(device)

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'defect_model_weights.pth')

    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    model.eval()

    _, val_loader, class_names = get_data_loaders(base_dir, batch_size=32)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    evaluate_model()