import os

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class DiseaseHead(nn.Module):
    def __init__(self, in_channels, hidden=0):
        super().__init__()
        self.net = self._build_net(in_channels, hidden)

    @staticmethod
    def _build_net(in_channels, hidden):
        if hidden and hidden > 0:
            return nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        return nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.net(x).view(-1)


def cell_embeddings(data, z):
    return z[torch.bitwise_not(data.y)]


def has_disease_labels(data):
    return hasattr(data, "disease_y") and data.disease_y is not None


def disease_loss(model, data, z):
    if not has_disease_labels(data):
        return z.new_tensor(0.0)
    logits = model.disease_logits(cell_embeddings(data, z))
    labels = data.disease_y.to(logits.device).float()
    mask = torch.bitwise_not(torch.isnan(labels))
    if mask.sum() == 0:
        return z.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def disease_score_frame(model, data, new_data):
    z = model.encode(data.x, data.edge_index)
    probs = model.disease_probabilities(cell_embeddings(data, z)).detach().cpu()
    df = pd.DataFrame({"cell_id": range(len(probs)), "disease_probability": probs.numpy()})
    df["cell_name"] = list(new_data.obs_names)
    df["cell_type"] = list(new_data.obs["clusters"])
    if has_disease_labels(data):
        df["disease_label"] = data.disease_y.detach().cpu().numpy()
    return df


def disease_weight_rows(model):
    rows = []
    for name, param in model.disease_head.named_parameters():
        values = param.detach().cpu().view(-1).numpy()
        rows.extend({"parameter": name, "index": i, "disease_weight": v} for i, v in enumerate(values))
    return rows


def disease_weight_frame(model):
    return pd.DataFrame(disease_weight_rows(model))


def save_disease_outputs(model, data, new_data, args):
    if not has_disease_labels(data):
        return
    model.eval()
    with torch.no_grad():
        scores = disease_score_frame(model, data, new_data)
    scores.to_csv(os.path.join(args.output_dir, f"{args.exp_name}_disease_scores.csv"), index=False)
    disease_weight_frame(model).to_csv(os.path.join(args.output_dir, f"{args.exp_name}_disease_head_weights.csv"), index=False)
