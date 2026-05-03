import torch
import torch.nn.functional as F
from load_biomedclip import CONTEXT_LENGTH


def build_text_features(model, tokenizer, class_names, templates, device):
    """Encode all (template, class) combinations and average-pool per class."""
    all_text_features = []
    with torch.no_grad():
        for name in class_names:
            prompts = [t.format(name) for t in templates]
            tokens = tokenizer(prompts, context_length=CONTEXT_LENGTH).to(device)
            feats = model.encode_text(tokens)  # (n_templates, d)
            feats = F.normalize(feats, dim=-1)
            feats = feats.mean(dim=0)           # average over templates
            feats = F.normalize(feats, dim=-1)
            all_text_features.append(feats)
    return torch.stack(all_text_features, dim=0)  # (n_classes, d)


def zero_shot_predict(model, preprocess, tokenizer, dataloader,
                      class_names, templates, device):
    """
    Run zero-shot classification over a dataloader that yields (images, labels).
    Returns (predictions, ground_truth) as CPU tensors.
    """
    text_features = build_text_features(
        model, tokenizer, class_names, templates, device
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits = image_features @ text_features.T  # (batch, n_classes)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)
