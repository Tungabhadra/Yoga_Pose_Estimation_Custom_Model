import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import onnxruntime

class YogaPoseNet(nn.Module):
    def __init__(self, num_classes):
        super(YogaPoseNet, self).__init__()
        # Use ResNet18 as base model
        self.model = models.resnet18(pretrained=True)
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.model(x)
        confidence = self.softmax(x)
        return x, confidence

class YogaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class YogaPoseClassifier:
    def __init__(self, data_dir, batch_size=32, num_epochs=21):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def prepare_data(self):
        image_paths = []
        labels = []
        self.classes = []
        
        for class_idx, class_name in enumerate(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_idx)
        
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        self.train_dataset = YogaDataset(X_train, y_train, self.transform)
        self.val_dataset = YogaDataset(X_val, y_val, self.val_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size
        )
        
        return len(self.classes)
    
    def train_model(self):
        num_classes = len(self.classes)
        self.model = YogaPoseNet(num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.1
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, confidence = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': train_loss/train_total,
                    'acc': 100.*train_correct/train_total
                })
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, confidence = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss/val_total
            val_acc = 100.*val_correct/val_total
            
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def export_onnx(self, output_path='yoga_pose_classifier.onnx'):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['class_scores', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'class_scores': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path}")
        
        class_mapping = {i: class_name for i, class_name in enumerate(self.classes)}
        with open('class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print("Class mapping saved to class_mapping.json")

# Example usage for inference with confidence scores
def inference_with_confidence(image_path, onnx_path='yoga_pose_classifier.onnx', class_mapping_path='class_mapping.json'):
    def prepare_image(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image).transpose(2, 0, 1)
        image = image / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
        return image.astype(np.float32)[None, ...]

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # Initialize ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path)

    # Prepare input image
    image = prepare_image(image_path)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(['class_scores', 'confidence'], {input_name: image})
    
    # Get predictions and confidence scores
    confidence_scores = outputs[1][0]
    predicted_class_idx = np.argmax(confidence_scores)
    predicted_class = class_mapping[str(predicted_class_idx)]
    confidence = confidence_scores[predicted_class_idx]

    # Get top 3 predictions with confidence scores
    top_3_indices = np.argsort(confidence_scores)[-3:][::-1]
    top_3_predictions = [
        {
            'pose': class_mapping[str(idx)],
            'confidence': float(confidence_scores[idx])
        }
        for idx in top_3_indices
    ]

    return {
        'top_prediction': {
            'pose': predicted_class,
            'confidence': float(confidence)
        },
        'top_3_predictions': top_3_predictions
    }

def main():
    # Set your dataset directory
    data_dir = "./yoga_dataset"
    
    # Initialize and train classifier
    classifier = YogaPoseClassifier(data_dir=data_dir)
    classifier.prepare_data()
    classifier.train_model()
    classifier.export_onnx()
    
    # Example inference
    # result = inference_with_confidence('./trial.jpg')
    # print("\nPrediction Results:")
    # print(f"Top prediction: {result['top_prediction']['pose']} "
    #       f"(Confidence: {result['top_prediction']['confidence']:.2%})")
    # print("\nTop 3 predictions:")
    # for pred in result['top_3_predictions']:
    #     print(f"{pred['pose']}: {pred['confidence']:.2%}")

if __name__ == "__main__":
    main()