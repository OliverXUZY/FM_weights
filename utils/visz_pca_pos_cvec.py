import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


n_plot_seq = 1
nrows, ncols = 2, 6


def plot(pos, cvec, plot_context=True, save_to=None):
    L, T, C = pos.shape

    if L == 13:
        selected_layers = range(L - 1)
    else:
        selected_layers = list(np.linspace(0, L - 2, num=12, dtype=int))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)

    for subplot_idx, layer_idx in enumerate(
        tqdm(selected_layers, desc="Layer progress")
    ):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        p = pos[layer_idx]

        # print("p shape: ", p.shape) # [197, 192]

        # apply PCA
        u, s, vt = np.linalg.svd(p)
        proj_mat = vt[:2, :].T

        # print("proj_mat shape: ", proj_mat.shape) # [192, 2]
        pc = p @ proj_mat

        # print("pc shape: ", pc.shape) # [197, 2]

        colors_blue = [plt.cm.Blues(x) for x in np.linspace(0.3, 1, T)]

        ax = axs[rdx, cdx]
        ax.scatter(pc[:, 0], pc[:, 1], c=colors_blue, label="pos")

        if plot_context:
            c = cvec[layer_idx, : n_plot_seq * T]
            # print("c shape: ", c.shape) # [197, 197, 192]
            pc2 = c @ proj_mat
            # print("pc2 shape: ", pc2.shape) # [197, 197, 2]
            colors_red = np.array(
                [plt.cm.Reds(x) for x in np.linspace(0.3, 1, T)] * n_plot_seq
            )
            ax.scatter(pc2[:, 0], pc2[:, 1], c=colors_red, alpha=0.1, label="c-vec")

        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=30)

    for ax in axs.ravel():
        ax.set_axis_off()
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot3D(pos, cvec, plot_context=True, save_to=None, remove_axis = True):
    nrows, ncols = 3, 4

    L, T, C = pos.shape

    if L == 13:
        selected_layers = range(L - 1)
    else:
        selected_layers = list(np.linspace(0, L - 2, num=12, dtype=int))


    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100, subplot_kw={'projection': '3d'})

    for subplot_idx, layer_idx in enumerate(
        tqdm(selected_layers, desc="Layer progress")
    ):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        p = pos[layer_idx]

        # print("p shape: ", p.shape) # [197, 192]

        # apply PCA
        u, s, vt = np.linalg.svd(p)
        proj_mat = vt[:3, :].T

        # print("proj_mat shape: ", proj_mat.shape) # [192, 3]
        pc = p @ proj_mat

        colors_blue = [plt.cm.Blues(x) for x in np.linspace(0.3, 1, T)]

        ax = axs[rdx, cdx]
        # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors_blue, label="pos")
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=0.8, label="pos")


        if plot_context:
            c = cvec[layer_idx, : n_plot_seq * T]
            # print("c shape: ", c.shape) # [197, 197, 192]
            pc2 = c @ proj_mat
            # print("pc2 shape: ", pc2.shape) # [197, 197, 2]
            colors_red = np.array(
                [plt.cm.Reds(x) for x in np.linspace(0.3, 1, T)] * n_plot_seq
            )
            ax.scatter(pc2[:, 0], pc2[:, 1], c=colors_red, alpha=0.1, label="c-vec")

        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=30)

    if remove_axis:
        for ax in axs.ravel():
            ax.set_axis_off()
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def determine_layout(num_plots):
    # You can define your own logic for how many rows and columns you want
    nrows = int(np.sqrt(num_plots))
    ncols = (num_plots + nrows - 1) // nrows  # Ensuring enough columns
    return nrows, ncols


def plot1D(pos, save_to=None, point_size = 8, largeLM = None):
    PC = {}
    L, T, C = pos.shape

    # if L == 13:
    #     selected_layers = range(L - 1)
    # else:
    #     selected_layers = list(np.linspace(0, L - 2, num=12, dtype=int))
    
    selected_layers = range(1,L)

    nrows, ncols = determine_layout(len(selected_layers))  # Define how you want to layout your subplots

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)

    for subplot_idx, layer_idx in enumerate(tqdm(selected_layers, desc="Layer progress")):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        p = pos[layer_idx]

        # apply PCA
        u, s, vt = np.linalg.svd(p)
        proj_mat = vt[:1, :].T  # Taking the first PC only
        PC[layer_idx] = proj_mat[:, 0]


        # colors_blue = [plt.cm.Blues(x) for x in np.linspace(0.3, 1, C)]

        ax = axs[rdx, cdx]
        # ax.scatter(pc[:, 0], np.zeros_like(pc[:, 0]), c=colors_blue, label="pos")  # Plotting on the first PC only
        x_axis = np.arange(1, proj_mat.shape[0] + 1)  # x-axis from 1 to number of data points
        ax.scatter(x_axis, proj_mat[:, 0], label="pos", s = point_size)  # Plotting raw values on the first PC
        ax.vlines(x_axis, ymin=0, ymax=proj_mat[:, 0], alpha=0.5)

        if largeLM is not None:
            # Draw vertical lines based on largeLM dictionary
            for index in largeLM.get(layer_idx, []):
                ax.axvline(x=index, color='r', linestyle='--', lw=0.5)  # Adding a vertical line

        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=30)

    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return PC




def plotLayerNorm(model, save_to=None, point_size = 8, norm_option = "attention", para_option = "weight"):
    weight = {}
    bias = {}
    selected_layers = range(12)

    nrows, ncols = determine_layout(len(selected_layers))  # Define how you want to layout your subplots

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)

    for subplot_idx, layer_idx in enumerate(selected_layers):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        if norm_option == "attention":
            gamma = model.blocks[layer_idx].norm1.weight
            beta = model.blocks[layer_idx].norm1.bias
        elif norm_option == "mlp":
            gamma = model.blocks[layer_idx].norm2.weight
            beta = model.blocks[layer_idx].norm2.bias
        else:
            raise NotImplementedError("not implemented yet!")

        

        gamma = gamma.detach().numpy()
        beta = beta.detach().numpy()

        weight[layer_idx+1] = gamma
        bias[layer_idx+1] = beta

        if para_option == "weight":
            para = gamma
        elif para_option == "bias":
            para = beta
        else:
            raise NotImplementedError("not implemented yet!")

        ax = axs[rdx, cdx]
        # ax.scatter(pc[:, 0], np.zeros_like(pc[:, 0]), c=colors_blue, label="pos")  # Plotting on the first PC only
        x_axis = np.arange(1, para.shape[0] + 1)  # x-axis from 1 to number of data points
        ax.scatter(x_axis, para, label="pos", s = point_size)  # Plotting raw values on the first PC
        ax.vlines(x_axis, ymin=0, ymax=para, alpha=0.5)

        ax.set_title(f"Layer: {layer_idx + 1}", weight="bold", fontsize=30)


    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return weight, bias
