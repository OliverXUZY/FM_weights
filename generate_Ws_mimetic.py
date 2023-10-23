import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", type=str,default="vit_base_patch16_224")
    parser.add_argument("--num_heads", type=int, default= 12)
    parser.add_argument("--num_blocks", type=int, default= 12)
    return parser.parse_args()
def main():
    args = parse_args()
    model_card = args.model_card
    num_heads = args.num_heads
    num_blocks = args.num_blocks

    model = timm.create_model(model_card, pretrained=True)
    state_dict = model.state_dict()

    # Initialize containers for all blocks' weights
    W_q_all, W_k_all, W_v_all, W_proj_all = [], [], [], []

    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}.attn."

        W_qkv = state_dict[prefix + "qkv.weight"].t()  # Transpose for splitting
        W_q, W_k, W_v = torch.chunk(W_qkv, 3, dim=1)
        W_q_all.append(torch.chunk(W_q, num_heads, dim=0))
        W_k_all.append(torch.chunk(W_k, num_heads, dim=0))
        W_v_all.append(torch.chunk(W_v, num_heads, dim=0))
        W_proj_all.append(torch.chunk(state_dict[prefix + "proj.weight"], num_heads, dim=0))

    matrix_q_k_all = [[torch.mm(W_q_all[block][head], W_k_all[block][head].t()).detach().numpy() 
                    for head in range(num_heads)] 
                    for block in range(num_blocks)]

    matrix_v_proj_sum_all = [sum([torch.mm(W_v_all[block][head], W_proj_all[block][head].t()).detach().numpy() 
                                for head in range(num_heads)]) 
                            for block in range(num_blocks)]
    
    # Visualization Function for W_q * W_k^T
    def visualize_qk_as_pdf(matrix_q_k_all, filename):
        nblocks, nheads = len(matrix_q_k_all), len(matrix_q_k_all[0])
        fig, axes = plt.subplots(nheads, nblocks, figsize=(20, 5/3*num_heads))
        
        for block_idx in range(nblocks):
            for head_idx in range(nheads):
                ax = axes[head_idx][block_idx]
                ax.imshow(matrix_q_k_all[block_idx][head_idx], cmap='viridis')
                ax.set_title(f"Block {block_idx+1}, Head {head_idx+1}")
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format="pdf", bbox_inches='tight')

    # visualize_qk_as_pdf(matrix_q_k_all, "vit_base_W_q_W_k_T_test.pdf")
    visualize_qk_as_pdf(matrix_q_k_all, "{}_W_q_W_k_T.pdf".format(model_card))


    # Visualization Function for Sum of W_v * W_proj
    def visualize_v_proj_sum_as_pdf(matrix_v_proj_sum_all, filename):
        fig, axes = plt.subplots(1, len(matrix_v_proj_sum_all), figsize=(20, 2))
        
        for block_idx, matrix in enumerate(matrix_v_proj_sum_all):
            ax = axes[block_idx]
            ax.imshow(matrix, cmap='viridis')
            ax.set_title(f"Block {block_idx+1}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format="pdf", bbox_inches='tight')

    visualize_v_proj_sum_as_pdf(matrix_v_proj_sum_all, "{}_W_v_W_proj_sum.pdf".format(model_card))


if __name__ == '__main__':
    main()