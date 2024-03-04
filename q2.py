import torch
from torch import nn
import torchvision.models as models
from utils import q2_get_train_loader, q2_get_test_loader, test_accuracy

def train_model():
    # Your code here: Reuse resnet18 as the model
    model = None
    # Your code here: Change the output layer so that we classify 2 classes

    # Prepare the model for training
    train_loader = q2_get_train_loader()
    train_dataset_size = len(train_loader.dataset)
    model.train()

    # Your code here: Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Your code here: Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Epoch (Configurable)
    number_of_epoch = 2

    # Start training (Hint: Follow Q1 and fill in the blanks)
    for epoch in range(1, number_of_epoch + 1):
        for batch_index, (inputs, labels) in enumerate(train_loader):
            # Your code here: Reset optimizer gradient
            
            # Your code here: Forward pass
            
            # Your code here: Calculate loss
            
            # Your code here: Back propagation to tune gradients of parameters to minimize loss
            
            # Your code here: Perform gradient descent

            # Optional: Show accuracy (if you copy from q1, no need to include "if...% 100..." check as we only have few batches)
            pass

    return model



if __name__ == "__main__":
    model = train_model()
    torch.save(model, 'q2.pth')
    print(f'Test dataset accuracy is : {test_accuracy(model, q2_get_test_loader()):.2f}%')