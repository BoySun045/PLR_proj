import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


def get_model(input_dim, neuron_hidden_layers, device, **kwargs):
    model = Model(input_dim, neuron_hidden_layers, device)
    return model


class Model(nn.Module):
    def __init__(self, input_dim, neuron_hidden_layers, device):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, neuron_hidden_layers[0])
        self.fc1 = nn.Linear(neuron_hidden_layers[0], neuron_hidden_layers[1])
        self.fc2 = nn.Linear(neuron_hidden_layers[1], neuron_hidden_layers[2])
        self.fc_out = nn.Linear(neuron_hidden_layers[2], 3)
        self.activation = nn.ReLU()
        self.image_encoder = torchvision.models.resnet34(pretrained=True)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.depth_encoder = Encoder(64)
        self.semantic_encoder = Encoder(64)
        self.normal_encoder = Encoder(64)
        self.device = device

    def forward(self, data_dict):
        image_encoded = self.image_encoder(data_dict["img_og"].to(self.device))
        depth_encoded = self.depth_encoder(data_dict["depth_pred"].to(self.device))
        semantic_encoded = self.semantic_encoder(data_dict["semantic_pred"].to(self.device))
        normal_encoded = self.normal_encoder(data_dict["normal_pred"].to(self.device))
        features = torch.cat((image_encoded, depth_encoded, semantic_encoded, normal_encoded), dim=1)
        out = self.fc_in(features)
        out = self.fc1(self.activation(out))
        out = self.fc2(self.activation(out))
        result = self.fc_out(self.activation(out))
        return result


class Encoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mobile_net_v2 = torchvision.models.mobilenet_v2()
        self.mobile_net_v2.classifier = torch.nn.Linear(1280, out_dim)

    def forward(self, input_image):
        out = self.mobile_net_v2(input_image)
        return out
