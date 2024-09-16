import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from norse.torch import LIFCell, LIFParameters, LICell

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 5

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Spiking Neural Network Model
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)    # First layer: Fully connected
        self.lif1 = LIFCell()                 # First spiking layer
        self.fc2 = nn.Linear(128, 64)         # Second layer: Fully connected
        self.lif2 = LIFCell()                 # Second spiking layer
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

# Model, loss function, and optimizer
model = SNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the SNN
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Testing the SNN
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# Main training loop
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)