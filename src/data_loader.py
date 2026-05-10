import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dir = os.path.join(data_dir, 'train', 'images')
    val_dir = os.path.join(data_dir, 'validation', 'images')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes

if __name__ == "__main__":

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    train_dl, val_dl, class_names = get_data_loaders(base_dir)
    print(f"Classes found: {class_names}")

    images, labels = next(iter(train_dl))
    print(f"Batch dimensions of images: {images.shape}")
    print(f"Dimensions of label batch: {labels.shape}")