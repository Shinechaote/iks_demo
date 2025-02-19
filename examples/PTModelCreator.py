import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# A multilayer perceptron (MLP) is a name for a modern feedforward neural network consisting of fully connected neurons with nonlinear activation functions.
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.view(-1, 784)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.view(-1, 784)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        print(f"Accuracy: {100*correct/total:.2f}%")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="./../SampleDataset/Train", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./../SampleDataset/Test", train=False, transform=transform, download=True)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=10e-3)

    train_model(model, train_loader, criterion, optimizer, epochs=5)
    evaluate_model(model, test_loader)
    torch.save(model, "torchMNISTModel.pth")