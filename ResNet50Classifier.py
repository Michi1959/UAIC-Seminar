import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from datetime import datetime


class ResNet50Classifier:
    def __init__(self, train_path, num_classes=10, batch_size=32, lr=0.001, trainable_layers=None, patience=5):
        self.train_path = train_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.trainable_layers = trainable_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        self.model = self._initialize_model()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.train_loader, self.val_loader = self._prepare_data()

    def _initialize_model(self):
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        if self.trainable_layers is None:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith('fc')
        else:
            for name, param in model.named_parameters():
                is_trainable = any(layer in name for layer in self.trainable_layers)
                param.requires_grad = is_trainable or name.startswith('fc')

        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _prepare_data(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(root=self.train_path, transform=train_transform)

        train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(self.train_loader)
        train_accuracy = 100 * correct_train / total_train
        return avg_train_loss, train_accuracy

    def validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return val_loss / len(self.val_loader), accuracy

    def train(self, num_epochs, excel_file_prefix='training_metrics', save_models_dir='saved_models', model_name='model'):
        metrics = []
        total_start_time = time.time()

        os.makedirs(save_models_dir, exist_ok=True)

        for epoch in range(num_epochs):
            if self.early_stop:
                print("Early stopping triggered")
                break

            epoch_start_time = time.time()
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # Save only the final model
                final_model_path = f"{save_models_dir}/{model_name}_final.pth"
                self.save_model(final_model_path)
                print(f"Final model saved at: {final_model_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch + 1}/{num_epochs}, Time: {epoch_duration:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

            self.scheduler.step()

            metrics.append({
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Train Accuracy (%)': train_accuracy,
                'Val Loss': val_loss,
                'Val Accuracy (%)': val_accuracy,
                'Time (seconds)': epoch_duration
            })

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"Total training time: {total_duration:.2f} seconds")

        metrics.append({
            'Epoch': 'Total',
            'Train Loss': '-',
            'Train Accuracy (%)': '-',
            'Val Loss': '-',
            'Val Accuracy (%)': '-',
            'Time (seconds)': total_duration
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = f"{excel_file_prefix}_{timestamp}.xlsx"

        df = pd.DataFrame(metrics)
        with pd.ExcelWriter(excel_file, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=model_name)
        print(f"Metrics saved to {excel_file}")

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'trainable_layers': self.trainable_layers
        }, filepath)


def main():
    train_path = r"D:\Seminar\Datenbanken\EuroSAT_RGB\train"

    classifier = ResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers=['layer4'])
    classifier.train(num_epochs=25, save_models_dir='saved_models', model_name='classifier1')

    classifier2 = ResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers=['layer3', 'layer4'])
    classifier2.train(num_epochs=25, save_models_dir='saved_models', model_name='classifier2')

    classifier3 = ResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers=['layer2', 'layer3', 'layer4'])
    classifier3.train(num_epochs=25, save_models_dir='saved_models', model_name='classifier3')

    classifier4 = ResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers=None)
    classifier4.train(num_epochs=25, save_models_dir='saved_models', model_name='classifier4_fc_only')

    classifier5 = ResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers='all')
    classifier5.train(num_epochs=25, save_models_dir='saved_models', model_name='classifier5_all_layers')

if __name__ == "__main__":
    main()
