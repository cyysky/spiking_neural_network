#SPIKING NEURAL NETWORK WEBCAM DIGIT RECOGNITION
#Author : Chong Yoe Yat
#Date : 2024SEP16
#Spiking Neural Network Train with webcam to recognize 1,2,3
#https://www.youtube.com/watch?v=X0qicnMrYSE

import torch
import torch.nn as nn
import norse.torch as norse
import cv2
import os
import numpy as np
from torchvision import transforms
import time

#Spiking Neural Network Model
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)    # First layer: Fully connected
        self.lif1 = norse.LIFCell()                 # First spiking layer
        self.fc2 = nn.Linear(128, 64)         # Second layer: Fully connected
        self.lif2 = norse.LIFCell()                 # Second spiking layer
        self.fc3 = nn.Linear(64, 10)          # Output layer for 10 classes (MNIST)
    
    def forward(self, x):
        # Flatten the image (28x28) to a vector of size 784
        x = x.view(-1, 28 * 28)
        
        # First layer: Linear + Spiking activation
        x = self.fc1(x)
        z1, _ = self.lif1(x)  # LIF Layer: Spiking computation
        
        # Second layer: Linear + Spiking activation
        x = self.fc2(z1)
        z2, _ = self.lif2(x)
        
        # Output layer (no spiking in the final layer)
        x = self.fc3(z2)
        return x

# Helper functions
def save_image(image, label, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f'{label}_{int(time.time())}.png')
    cv2.imwrite(filename, image)
    print(f'Saved {filename}')

def load_images_from_folder(folder='data'):
    data, labels = [], []
    
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            # Assuming filenames are saved as "<label>_<timestamp>.png"
            label = int(filename.split('_')[0])
            
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 
            data.append(image)
            labels.append(label)
        print(f"Loaded {len(data)} images from {folder}")
    else:
        print("No existing data found.")
    
    return data, labels

        
def load_model():
    model = SNN()
    if os.path.exists('model_state_dict.pth'):
        model.load_state_dict(torch.load('model_state_dict.pth'))
        print("Model loaded from 'model_state_dict.pth'")
    else:
        print("No pre-trained model found. Starting fresh.")
    return model

# non batch training
def train_model(model, data, labels, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    
    for epoch in range(epochs):
        for i, (img, label) in enumerate(zip(data, labels)):
            img = torch.Tensor(img).unsqueeze(0)  # Add batch dimension
            label = torch.tensor([label])
            
            # Forward pass
            output = model(img)
            loss = criterion(output, label)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# training with batch
def train_model(model, data, labels, batch_size=5, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    
    dataset_size = len(data)
    for epoch in range(epochs):
        for i in range(0, dataset_size, batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Convert to tensors and create batches
            batch_data = torch.Tensor(batch_data)
            batch_labels = torch.Tensor(batch_labels).long()  # Convert labels to LongTensor
            
            # Forward pass
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}/{epochs}, Batch {i//batch_size+1}/{(dataset_size+batch_size-1)//batch_size}, Loss: {loss.item()}')
        scheduler.step()

def predict(model, image):
    model.eval()
    with torch.no_grad():
        img = torch.Tensor(image).unsqueeze(0)  # Add batch dimension
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def main():
    model = load_model()
    data, labels = load_images_from_folder()  # Load previous data
    #train_model(model, data, labels)
    cap = cv2.VideoCapture(0)
    print("Press 's' to save an image and start training, 'r' to predict, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        key = cv2.waitKey(1)
                
        if key in range(48, 58):  # Keys '0' to '9'
            label = key - 48  # Convert ASCII to digit
            image_resized = cv2.resize(gray, (28, 28))  # Resize to 28x28
            save_image(image_resized, label)
            data.append(image_resized)  # Append new data
            labels.append(label)
            
        elif key == ord('t'):  # Predict
            train_model(model, data, labels,epochs=10)
            
        elif key == ord('r'):  # Predict
            image_resized = cv2.resize(gray, (28, 28))
            pred = predict(model, image_resized)
            print(f'Predicted digit: {pred}')
        
        elif key == ord('q'):  # Quit and save model
            torch.save(model.state_dict(), 'model_state_dict.pth')
            print("Model saved as 'model_state_dict.pth'")
            break
            
        image_resized = cv2.resize(gray, (28, 28))
        pred = predict(model, image_resized)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        gray = cv2.putText(gray, str(pred), org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Webcam', gray)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
