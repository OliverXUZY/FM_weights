import torch
import timm
import random
import argparse
import os

from src import get_sets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_384")
    parser.add_argument("--data_root", type=str, default="/Users/zyxu/Documents/py/datasets")
    parser.add_argument("--output_path", type=str, default="output")

    return parser.parse_args()

def _add_hooks(model):
  # Assuming model is VisionTransformer instance
  intermediate_outputs = []

  # Hook function
  def pre_forward_hook(module, input):
    intermediate_outputs.append(input[0].clone())

  model.blocks[0].attn.register_forward_pre_hook(pre_forward_hook)

  # Hook function to capture outputs
  def forward_hook(module, input, output):
    intermediate_outputs.append(output.clone())

  # Register the hook to capture the output after the initial patch embedding
  # model.patch_embed.register_forward_hook(forward_hook)

  # Register the forward hook after each block
  for block in model.blocks:
    block.register_forward_hook(forward_hook)
  
  return intermediate_outputs


def get_random_batch(dataset, batch_size, rnd):
    """
    Get a random batch from the dataset without using DataLoader.
    
    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset object.
    - batch_size (int): The size of the batch to retrieve.
    - seed_value (int): Value of the seed for reproducibility.
    
    Returns:
    - list: A batch of data points randomly sampled from the dataset.
    """
    
    
    # Randomly select indices for the batch
    indices = rnd.sample(range(len(dataset)), batch_size)
    
    # Get the data points corresponding to these indices
    batch = [dataset[i] for i in indices]
    
    # If your dataset returns (data, target) pairs, you might want to separate them
    data, targets = zip(*batch)
    
    # Convert data and targets to tensors
    data_tensor = torch.stack(data)
    targets_tensor = torch.tensor(targets)
    
    return data_tensor, targets_tensor


def main():
  args = parse_args()
  model = timm.create_model(args.model, img_size=224,pretrained=True)

  intermediate_outputs = _add_hooks(model)

  '''
  for debug:
  # Now run your model with some data
  input_tensor = torch.rand(64, 3, 224, 224)
  model(input_tensor)

  # The intermediate_outputs list should now contain the outputs before each attention layer
  print(len(intermediate_outputs))  # Number of blocks with attention layers
  print(intermediate_outputs[0].shape)  # Should be [64, 196, 192] or similar
  print(intermediate_outputs[0])


  for dataset_name in ["Cifar10", "Omniglot", "STL10"]:
    sets[dataset_name] = get_sets(dataset_name)
    print("========= name is: ", dataset_name)
    print(sets[dataset_name][0][0].shape)
    print(sets[dataset_name][0][1])
  
  '''

  sets = {}
  
  # Set the random seed
  rnd = random.Random()
  rnd.seed(42)


  batch = []
  names = ["Cifar10", "DTD", "STL10", "tieredImageNet"]
  for dataset_name in names:
    # print("========= name is: ", dataset_name)
    sets[dataset_name] = get_sets(dataset_name, data_root = args.data_root)
    data_tensor, targets_tensor = get_random_batch(sets[dataset_name], batch_size=64,rnd = rnd)
    
    batch.append(data_tensor)
  
  batch = torch.concat(batch)
  # print("batch.shape: ", batch.shape)

  model(batch)

  # print(intermediate_outputs[0].dtype)

  intermediate_outputs = torch.stack(intermediate_outputs)

  # print(intermediate_outputs.shape)

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  torch.save(intermediate_outputs, 'output/4sets_intermediate_outputs.pth')


if __name__ == "__main__":
  main()

    

    
    