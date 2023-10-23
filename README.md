# FM_weights

## Environment setup
```
conda create --name FM_weights python==3.9
conda activate FM_weights
pip install -r requirements.txt
```
## Get Embeddings 
### Generate Embeddings
**Generate $\{h_{c,t}\}$**
```
python generate_pos_ctx.py --data_root /Users/zyxu/Documents/py/datasets --output_path output
```

**Datasets**
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))

**Local Data Directory**
Download and extract data into `data_root` like below, in my case `data_root = /Users/zyxu/Documents/py/datasets`
```
tiered_imagenet
├── tiered-imagenet-kwon
│   ├── test_images.npz
│   ├── test_labels.pkl
│   ├── train_images.npz
│   ├── train_labels.pkl
│   ├── val_images.npz
│   └── val_labels.pkl

```
### Load Embeddings
Alternatively, you can download embeddings into `output/` from here: [4sets_intermediate_outputs.pth](https://drive.google.com/file/d/1ozlyHvSreweNpRQsDnsvdu5MiMkm_ZrS/view?usp=share_link)

## Reproducibility
After getting the embeddings `output/4sets_intermediate_outputs.pth`. `notebook/uncover/vis.ipynb` contains visulizations and `notebook/uncover/rank_compute.ipynb` contains rank computation.
