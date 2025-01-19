import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from datetime import datetime
from CustomResNet import ResNet50


class ScratchResNet50Classifier:
    def __init__(self, train_path, num_classes=10, batch_size=32, lr=0.001, trainable_layers=None, patience=5, early_stop_enabled=True):
        self.train_path = train_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.trainable_layers = trainable_layers
        self.early_stop_enabled = early_stop_enabled  # Early stopping flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        self.model = self._initialize_model()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.train_loader, self.val_loader = self._prepare_data()


    def _initialize_model(self):
        model = ResNet50(num_classes=self.num_classes)

        if self.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        elif self.trainable_layers:
            for name, param in model.named_parameters():
                param.requires_grad = any(layer in name for layer in self.trainable_layers)
        else:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

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

    def train(self, num_epochs, save_models_dir='saved_models', model_name='scratch_model',
              excel_file_prefix='training_metrics'):
        os.makedirs(save_models_dir, exist_ok=True)
        metrics = []
        total_start_time = time.time()

        for epoch in range(num_epochs):
            if self.early_stop_enabled and self.early_stop:
                print("Early stopping triggered")
                break

            epoch_start_time = time.time()
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                final_model_path = f"{save_models_dir}/{model_name}_final.pth"
                self.save_model(final_model_path)
                print(f"Final model saved at: {final_model_path}")
            else:
                self.counter += 1
                if self.early_stop_enabled and self.counter >= self.patience:
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

        total_duration = time.time() - total_start_time
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
        df.to_excel(excel_file, index=False)
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

    # Classifier 6 with early stopping
    classifier_6 = ScratchResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers='all', patience=5, early_stop_enabled=True)
    classifier_6.train(num_epochs=50, save_models_dir='saved_models', model_name='classifier6_scratchmodel')

    # Classifier 7 without early stopping
    classifier_7 = ScratchResNet50Classifier(train_path=train_path, num_classes=10, trainable_layers='all', patience=5, early_stop_enabled=False)
    classifier_7.train(num_epochs=50, save_models_dir='saved_models', model_name='classifier7_scratchmodel')

if __name__ == "__main__":
    main()
