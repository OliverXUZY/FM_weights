import torch
import torchvision
import torchvision.transforms as transforms
from .tiered_imagenet import TieredImageNet




img_size = 224
def vanilla_transform(statistics):
  return transforms.Compose([
                          transforms.Resize((img_size, img_size)),
                          transforms.ToTensor(),
                          transforms.Normalize(**statistics)
                          ])

def get_sets(name = "Cifar10", data_root = "/Users/zyxu/Documents/py/datasets"):
  if name == "Cifar10":
    statistics = {'mean': [0.4914, 0.4822, 0.4465], 
            'std':  [0.247, 0.243, 0.261]}
    transform = vanilla_transform(statistics)
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, 
                                        transform=transform
                                        )
    
  if name == "tieredImageNet":
    root = f"{data_root}/tiered_imagenet/tiered-imagenet-kwon"
    statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    transform = vanilla_transform(statistics)
    trainset = TieredImageNet(root = root, split='train', transform=transform)

  if name == "Omniglot":
    statistics = {'mean': [0.],
                  'std':  [1.]}
    transform = vanilla_transform(statistics)
    trainset = torchvision.datasets.Omniglot(root=data_root, 
                                          background=True,
                                          download=True, 
                                          transform=transform
                                          )
  if name == "STL10":
    statistics = {'mean': [0., 0., 0.],
                  'std':  [1., 1., 1.]}
    transform = vanilla_transform(statistics)
    trainset = torchvision.datasets.STL10(root=data_root, 
                                          split="train",
                                          download=True, 
                                          transform=transform
                                          )
  if name == "DTD":
    statistics = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}
    transform = vanilla_transform(statistics)
    trainset = torchvision.datasets.DTD(root=data_root, 
                                          download=True, 
                                          transform=transform
                                          )

  return trainset
