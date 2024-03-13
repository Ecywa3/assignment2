import torch
from torch import nn
import torchvision.models as models
from utils import q2_get_train_loader, q2_get_test_loader, test_accuracy

def train_model():
    # Your code here: Reuse resnet18 as the model
    model = models.resnet18(pretrained=True)
    # Your code here: Change the output layer so that we classify 2 classes

    # Prepare the model for training
    train_loader = q2_get_train_loader()
    train_dataset_size = len(train_loader.dataset)
    model.train()

    # Your code here: Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Your code here: Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # Epoch (Configurable)
    number_of_epoch = 5

    # Start training (Hint: Follow Q1 and fill in the blanks)
    for epoch in range(1, number_of_epoch + 1):
        correct_guess = 0
        total_guess = 0
        for batch_index, (inputs, labels) in enumerate(train_loader):
            # Your code here: Reset optimizer gradient
            optimizer.zero_grad()
            # Your code here: Forward pass
            labels_predicted = model(inputs)
            # Your code here: Calculate loss
            loss = loss_fn(labels_predicted, labels)
            # Your code here: Back propagation to tune gradients of parameters to minimize loss
            loss.backward()
            # Your code here: Perform gradient descent
            optimizer.step()
            # Optional: Show accuracy (if you copy from q1, no need to include "if...% 100..." check as we only have few batches)
            loss, current = loss.item(), (batch_index + 1) * len(inputs)
            # Record accuracy
            correct_guess += sum([1 for actual, predicted in zip(labels.tolist(), [l.argmax() for l in labels_predicted]) if actual==predicted])
            total_guess += len(inputs)
            accuracy = correct_guess / total_guess * 100.0
            print(f"[Epoch {epoch}][Batch {batch_index + 1}] Train Accuracy: {accuracy:3.2f}% Loss: {loss:3.4f}  [{current:5d}/{train_dataset_size:5d}]")

    return model


if __name__ == "__main__":
    model = train_model()
    torch.save(model, 'q2.pth')
    print(f'Train dataset accuracy is : {test_accuracy(model, q2_get_train_loader()):.2f}%')
    print(f'Test dataset accuracy is : {test_accuracy(model, q2_get_test_loader()):.2f}%')