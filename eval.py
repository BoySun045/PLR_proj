import torch
import yaml
import os
from src.trainer import Trainer
from src.model import get_model
from src.dataset import get_dataset
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-task', help='evaluation configuration')
args = parser.parse_args()
eval_config = args.task
config_path = './configs/config.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_workers = cfg['train']['num_workers']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate trainer
model = get_model(input_dim=1192, neuron_hidden_layers=[50, 25, 10], device=device)
model.to(device)
test_dataset = get_dataset('./test', "test_demo")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['test']['batch_size'], num_workers=num_workers, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if eval_config == "depth":      
    model_path = '../weights_in_all/attention/model_d.pt'
    out_img_path = 'test_d.png'
else:
    model_path = '../weights_in_all/attention/model.pt'
    out_img_path = 'test_s.png'

model.load_state_dict(torch.load(model_path))
trainer = Trainer(model, optimizer, device)

# test image
mask, error_estimate = trainer.test(test_loader)
# generate evaluation image
for test_idx in tqdm(range(len(mask))):
    data_path = os.path.join('./test', f'{test_idx}')
    img_og = Image.open(os.path.join(data_path, "img_og.png"))
    semantic = Image.open(os.path.join(data_path, "semantic.png")).convert('RGB')
    depth_pred = Image.open(os.path.join(data_path, "depth_pred_.png")).convert('RGB')
    depth_uncert = Image.open(os.path.join(data_path, "depth_uncertainty.png")).convert('RGB')
    semantic_uncert = Image.open(os.path.join(data_path, "semantic_uncertainty.png")).convert('RGB')
    normal = Image.open(os.path.join(data_path, "normal_pred.png")).convert('RGB')
    normal_uncert = Image.open(os.path.join(data_path, "normal_uncert.png")).convert('RGB')
    error = 
    fig = plt.figure(figsize=(40., 40.))

    plt.subplot(211)
    plt.axis('off')
    plt.title("Input Image", fontsize=100)
    plt.imshow(img_og)
    grid = ImageGrid(fig, 212,  # similar to subplot(111)
                     nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    plot_tiltes = ['Semantic', 'Depth', 'Normal']

    for idx, (ax, im) in enumerate(zip(grid, [semantic, depth_pred, normal, semantic_uncert, depth_uncert, normal_uncert])):
        # Iterating over the grid returns the Axes.
        if idx <= 2:
            depth_weight = mask[test_idx][0, idx]
            # sem_weight = mask[test_idx][0, idx + 3]
            ax.set_title(plot_tiltes[idx] + f' {depth_weight:.3f}', fontsize=100)
            # ax.set_title(plot_tiltes[idx] + f' weight {a}', fontsize=40)
        ax.set_axis_off()
        ax.imshow(im)
    # plt.show()
    plt.savefig(os.path.join(data_path, out_img_path), bbox_inches='tight')


