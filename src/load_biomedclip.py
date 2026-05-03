import torch
from open_clip import create_model_from_pretrained, get_tokenizer

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256


def load_model():
    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


if __name__ == "__main__":
    model, preprocess, tokenizer, device = load_model()
    print(f"Loaded BiomedCLIP on {device}")
    tokens = tokenizer(["a biomedical image"], context_length=CONTEXT_LENGTH).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    print(f"Text feature shape: {text_features.shape}")
