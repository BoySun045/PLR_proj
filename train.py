import torch
import yaml
import os
from PIL import Image
from src.trainer import Trainer
from src.model import get_model
from src.dataset import get_dataset
from tqdm import tqdm
import torchvision
import datetime
from torch.utils.tensorboard import SummaryWriter

# config parameters

config_path = './configs/config.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
epochs = cfg['train']['epochs']
batch_size = cfg['train']['batch_size']
num_workers = cfg['train']['num_workers']
dataset_path = cfg['train']['dataset_path']
out_dir = cfg['train']['out_dir']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create summary writer for tensorboard
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_logger = SummaryWriter(os.path.join(out_dir, 'logs', 'train', log_dir))
val_logger = SummaryWriter(os.path.join(out_dir, 'logs', 'val', log_dir))
# generate model
model = get_model(input_dim=1192, neuron_hidden_layers=[50, 25, 10], device=device)
model.to(device)

# generate dataset
dataset = get_dataset(dataset_path, "train")
# single-length test loader
split_len = [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)]
train_dataset, val_dataset = torch.utils.data.random_split(dataset, split_len)

# generate dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], num_workers=num_workers, shuffle=False)

# generate trainer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(model, optimizer, device)


for epoch in tqdm(range(epochs)):
    loss = []
    for batch in train_loader:
        loss.append(trainer.train_one_epoch(batch))

    average_loss = sum(loss)/len(loss)
    train_logger.add_scalar('loss', average_loss, epoch)
    val_loss = trainer.eval(val_loader)
    val_logger.add_scalar('loss', val_loss, epoch)


torch.save(model.state_dict(), 'model.pt')




