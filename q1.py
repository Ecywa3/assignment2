import torch
from torch import nn
from utils import q1_get_train_loader, q1_get_test_loader, test_accuracy

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 RGB channels => 1 channel, 32*32 pixels
        input_size = 1024*3 
        # 10 outputs
        output_size = len(CLASSES)
        # Configurable: Define our network
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = x.view(-1, 32*32*3) 
        return self.layers(x)

def train_model():
    model = NeuralNetwork()
    # Prepare the model for training
    train_loader = q1_get_train_loader()
    train_dataset_size = len(train_loader.dataset)
    model.train()
    # Configurable: Learning Rate
    learning_rate = 0.01
    # Loss function: We use Cross Entropy Loss for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    # Configurable: Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Configurable: Epoch
    number_of_epoch = 2

    # Start training
    for epoch in range(1, number_of_epoch + 1):
        correct_guess = 0
        total_guess = 0
        for batch_index, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels_predicted = model(inputs)
            loss = loss_fn(labels_predicted, labels)
            loss.backward()
            optimizer.step()
            # Record accuracy
            correct_guess += sum([1 for actual, predicted in zip(labels.tolist(), [l.argmax() for l in labels_predicted]) if actual==predicted])
            total_guess += len(inputs)
            # Show progress every 100 batches
            if (batch_index+1) % 100 ==0:
                loss, current = loss.item(), (batch_index+1) * len(inputs)
                accuracy = correct_guess/total_guess*100.0
                print(f"[Epoch {epoch}][Batch {batch_index+1}] Train Accuracy: {accuracy:3.2f}% Loss: {loss:3.4f}  [{current:5d}/{train_dataset_size:5d}]")
    return model

            
if __name__ == "__main__":
    model = train_model()
    torch.save(model, 'q1.pth')
    print(f'Test dataset accuracy is : {test_accuracy(model, q1_get_test_loader()):.2f}%')