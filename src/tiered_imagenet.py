import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class TieredImageNet(Dataset):
  def __init__(self, root, split='train', transform=None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(TieredImageNet, self).__init__()
    
    split_dict = {'train': 'train',         # standard train
                  'val': 'train_phase_val', # standard val
                  'meta-train': 'train',    # meta-train
                  'meta-val': 'val',        # meta-val
                  'meta-test': 'test',      # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root, split_tag + '_images.npz')
    # print(split_file)
    label_file = os.path.join(root, split_tag + '_labels.pkl')
    assert os.path.isfile(split_file)
    assert os.path.isfile(label_file)
    data = np.load(split_file, allow_pickle=True)['images']
    data = data[:, :, :, ::-1]
    with open(label_file, 'rb') as f:
      label = pickle.load(f)['labels']

    data = [Image.fromarray(x) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])

    self.root = root
    self.split_tag = split_tag
    
    self.data = data
    self.label = new_label
    self.n_class = len(label_key)

    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])              # [V, C, H, W]
    label = self.label[index]
    return image, label