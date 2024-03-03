import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import ssl
import logging
ssl._create_default_https_context = ssl._create_unverified_context  # This fixes SSL issues with dataset download

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class AntBeeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label




def q1_get_train_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    return train_loader

def q1_get_test_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    return test_loader


def q2_get_train_loader():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = AntBeeDataset('./hymenoptera_data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader

def q2_get_test_loader():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = AntBeeDataset('./hymenoptera_data/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

def get_device():
    if torch.cuda.is_available():
        logger.info('[Windows] Using CUDA')
        return torch.device('cuda:0')  # Windows/Old Apple: NVIDIA CUDA
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info('[Apple Silicon] Using MPS')
        return torch.device('mps')     # New Apple: Metal Performance Shaders
    logger.info('[Default] Using CPU')
    return torch.device('cpu')         # Default


def test_accuracy(model, loader):
    correct_guess, total_guess = 0, 0
    model.eval()
    with torch.no_grad():
        for (images, labels) in loader:
            labels_predicted = model(images)
            correct_guess += sum([1 for actual, predicted in zip(labels.tolist(), [l.argmax() for l in labels_predicted]) if actual==predicted])
            total_guess += len(images)
    return correct_guess/total_guess*100