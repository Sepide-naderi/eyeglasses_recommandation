import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---- Define Model Architectures ----

class ResNet18_Faceshape(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18_Faceshape, self).__init__()
        self.base_model = models.resnet18(pretrained=True)

        # Freeze base layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class FaceShapeCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FaceShapeCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (224x224) -> (224x224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (224x224) -> (112x112)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (112x112) -> (56x56)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (56x56) -> (28x28)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (28x28) -> (14x14)

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))  # (14x14) -> (1x1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # (256, 1, 1) -> (256)
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---- Load Model ----

def load_model(model_choice):
    if model_choice == "FaceshapeCNN":
        model = FaceShapeCNN(num_classes=5)
        model.load_state_dict(torch.load('model_epoch_100.pkt', map_location='cpu'))
    else:  # ResNet
        model = ResNet18_Faceshape(num_classes=5)
        model.load_state_dict(torch.load('model_resnet_epoch_42.pth', map_location='cpu'))

    model.eval()
    return model

# ---- Image Preprocessing ----

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # or IMAGE_SIZE if defined globally
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ---- Predict & Recommend ----

def predict_faceshape(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ['heart', 'long', 'oval', 'round', 'square']
        return class_names[predicted.item()]

def recommend_glasses(faceshape):
    recommendations = {
        'heart': 'Bottom-heavy frames',
        'long': 'Cat-eye or oval frames',
        'oval': 'Almost any frame shape',
        'round': 'Square or Rectangular glasses',
        'square': 'Round or Oval glasses'
    }
    return recommendations.get(faceshape, "No recommendation available")

st.title("Face Shape & Glasses Recommender")
st.write("Upload your photo and choose a model to analyze your face shape.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose model", ["FaceshapeCNN", "ResNet"])

if uploaded_file and model_choice:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model(model_choice)
    image_tensor = preprocess_image(image)

    faceshape = predict_faceshape(model, image_tensor)
    glasses = recommend_glasses(faceshape)

    st.success(f"**Predicted Face Shape:** {faceshape}")
    st.info(f"**Recommended Glasses:** {glasses}")
