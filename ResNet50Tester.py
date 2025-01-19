import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import pandas as pd
import time


class ResNet50Tester:
    def __init__(self, dataset_path, batch_size=32):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_loader = self._prepare_test_data()
        self.image_paths = self._get_image_paths()

    def _prepare_test_data(self):
        test_dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def evaluate_pretrained_models(self, model_paths):
        for model_path in model_paths:
            print(f"Evaluating pretrained model: {model_path}")
            model = self._load_pretrained_model(model_path)
            model.eval()
            model.to(self.device)

            metrics = []
            correct = 0
            total = 0
            start_time = time.time()

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    max_probs, predicted = probabilities.max(1)

                    for i in range(images.size(0)):
                        image_path = self.image_paths[batch_idx * self.batch_size + i]
                        true_label = labels[i].item()
                        predicted_label = predicted[i].item()
                        confidence = max_probs[i].item() * 100
                        class_name = self.test_loader.dataset.classes[predicted_label]

                        metrics.append({
                            'Image Path': image_path,
                            'True Class': self.test_loader.dataset.classes[true_label],
                            'Predicted Class': class_name,
                            'Confidence (%)': confidence
                        })

                        if predicted_label == true_label:
                            correct += 1
                        total += 1

            end_time = time.time()
            accuracy = 100 * correct / total
            duration = end_time - start_time

            print(f"Model: {model_path}, Test Accuracy: {accuracy:.2f}%, Time: {duration:.2f}s")

            # Save to Excel file
            excel_filename = f"{os.path.splitext(os.path.basename(model_path))[0]}_evaluation.xlsx"
            df = pd.DataFrame(metrics)
            df['Test Accuracy (%)'] = accuracy
            df['Evaluation Time (seconds)'] = duration
            df.to_excel(excel_filename, index=False)
            print(f"Evaluation results saved to {excel_filename}")

    def _load_pretrained_model(self, model_path):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(self.test_loader.dataset.classes))
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# Usage Example
def main():
    dataset_path = r"D:\\Seminar\\Datenbanken\\EuroSAT_RGB\\test"
    model_paths = [
        'saved_models/classifier1_final.pth',
        'saved_models/classifier2_final.pth',
        'saved_models/classifier3_final.pth',
        'saved_models/classifier4_fc_only_final.pth',
        'saved_models/classifier5_all_layers_final.pth',
        'saved_models/classifier6_scratchmodel_final.pth',
        'saved_models/classifier7_scratchmodel_final.pth',
    ]
    tester = ResNet50Tester(dataset_path=dataset_path)
    tester.evaluate_pretrained_models(model_paths=model_paths)


if __name__ == "__main__":
    main()
