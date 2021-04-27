import torch.nn as nn
import torch


class Trainer(object):
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = nn.MSELoss()

    def train_one_epoch(self, train_data):
        self.model.train()
        mask_output = self.model(train_data)
        loss = self.compute_loss(mask_output, train_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, test_loader):
        self.model.eval()
        mask_output = []
        error_estimate = []
        for test_batch in test_loader:
            depth_uncert = test_batch["depth_uncert"].to(self.device)
            semantic_uncert = test_batch["semantic_uncert"].to(self.device)
            normal_uncert = test_batch["normal_uncert"].to(self.device)
            weight = self.model(test_batch)
            # round the output
            n_digits = 4
            weight = (weight * 10**n_digits).round() / (10**n_digits)
            mask_output.append(weight)
            error = weight[:, 0].unsqueeze(0) * semantic_uncert.T + weight[:, 1].unsqueeze(0) * \
            depth_uncert.T + weight[:, 2].unsqueeze(0) * normal_uncert.T
            error_estimate.append(error)
            # loss = self.compute_loss(mask_output, test_batch).item()
        return mask_output, error_estimate


    def eval(self, val_loader):
        self.model.eval()
        val_loss = []
        for val_batch in val_loader:
            mask_output = self.model(val_batch)
            loss = self.compute_loss(mask_output, val_batch).item()
            val_loss.append(loss)

        return sum(val_loss)/len(val_loss)
        

    def compute_loss(self, weight, data, config="image_wise", acc_config="depth_rse"):
        depth_acc = data["depth_rse"].to(self.device)
        semantic_acc = data["semantic_ce"].to(self.device)

        depth_uncert = data["depth_uncert"].to(self.device)
        semantic_uncert = data["semantic_uncert"].to(self.device)
        normal_uncert = data["normal_uncert"].to(self.device)

        if config == "image_wise":
            # acc_estimate = weight[:, 0].unsqueeze(0) * semantic_uncert.T + weight[:, 1].unsqueeze(0) * depth_uncert.T
            acc_estimate = weight[:, 0].unsqueeze(0) * semantic_uncert.T + weight[:, 1].unsqueeze(0) * \
            depth_uncert.T + weight[:, 2].unsqueeze(0) * normal_uncert.T
            depth_loss = self.loss(acc_estimate, semantic_acc.T)

            # acc_estimate_sem = weight[:, 3].unsqueeze(0) * semantic_uncert.T + weight[:, 4].unsqueeze(0) * \
            # depth_uncert.T + weight[:, 5].unsqueeze(0) * normal_uncert.T
            # sem_loss = self.loss(acc_estimate_sem, semantic_acc.T)

            # loss = depth_loss + sem_loss
            return depth_loss
