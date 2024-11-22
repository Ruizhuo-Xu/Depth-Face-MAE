import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.img_dataset import ImgDataset

lock3dface_subsets = {"NU": 0, "FE": 1, "PS": 2, "OC": 3, "TM": 4}
def extract_gallery_features(model, gallery_path, device):
    gallery = ImgDataset(
        gallery_path, split='gallery', test_mode=True
    )
    imgs, labels = [], []
    for data in gallery:
        img, label, infos = data
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs, dim=0).to(device)
    labels = torch.stack(labels, dim=0).to(device)
    model.eval()
    with torch.no_grad():
        feats = model(imgs, only_return_feats=True)
    return feats, labels

def evaluation(prob_feats, prob_labels, gallery_feats,
               gallery_labels, metrics, is_lock3dface=False, **kwargs):
    prob_feats = F.normalize(prob_feats, p=2, dim=-1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=-1)
    sim = prob_feats @ gallery_feats.T
    max_idxs = torch.argmax(sim, dim=-1)
    preds = gallery_labels[max_idxs]
    results = {}
    total_acc = metrics["total_accuracy"](preds, prob_labels)
    results["total_accuracy"] = total_acc
    if is_lock3dface:
        infos = kwargs["img_infos"]
        subset_tags = infos["val_subset_label"]
        for k, v in lock3dface_subsets.items():
            subset_idxs = (subset_tags == v)
            if subset_idxs.any():
                subset_preds = preds[subset_idxs]
                subset_labels = prob_labels[subset_idxs]
                subset_acc = metrics[f"{k}_accuracy"](subset_preds, subset_labels)
                results[f"{k}_accuracy"] = subset_acc
            else:
                results[f"{k}_accuracy"] = -1
    return results