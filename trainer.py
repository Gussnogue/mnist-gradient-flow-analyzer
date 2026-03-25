import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class DeepNetwork(nn.Module):
    def __init__(self, activation='relu', depth=10, hidden_size=256):
        super().__init__()
        layers = []
        for i in range(depth):
            in_features = 28*28 if i == 0 else hidden_size
            out_features = hidden_size
            layers.append(nn.Linear(in_features, out_features))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
        # Camada de saída: 10 classes (MNIST)
        layers.append(nn.Linear(hidden_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def train_model(activation, depth, hidden_size, learning_rate, num_epochs=20, batch_size=64):
    # Carrega dados MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepNetwork(activation, depth, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    gradient_stats = []  # cada elemento: {'layer_grad_means': [], 'all_gradients': []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Coletar gradientes por camada para o primeiro batch de cada época (exemplo)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Coletar gradientes apenas no primeiro batch
            if batch_idx == 0:
                grad_means = []
                all_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.detach().cpu().numpy().flatten()
                        all_grads.append(grad)
                        grad_means.append(np.mean(np.abs(grad)))
                gradient_stats.append({
                    'layer_grad_means': grad_means,
                    'all_gradients': all_grads,
                    'epoch': epoch
                })
            if batch_idx >= 1:
                break  # usar apenas um batch para demonstração (acelera)

        losses.append(epoch_loss / (batch_idx+1))

    # Extrair nomes das camadas para visualização
    layer_names = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)

    return losses, gradient_stats, layer_names

