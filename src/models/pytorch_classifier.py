"""
SmartTriage PyTorch Deep Learning Classifier from First Principles.
Implements custom Vocabulary, PyTorch Dataset, DataLoader collation,
nn.Module architecture (Embedding + Pooling + MLP), and explicit training loop.
"""

from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
PYTORCH_MODEL_FILE = MODEL_REGISTRY_DIR / "pytorch_model.pt"
PYTORCH_VOCAB_FILE = MODEL_REGISTRY_DIR / "pytorch_vocab.json"


class Vocabulary:
    """Maps tokens to unique integers, handling <PAD> and <UNK>."""

    def __init__(self, min_freq: int = 1):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.min_freq = min_freq
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_token = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, texts: list[str]):
        token_counts = {}
        for text in texts:
            for token in text.lower().split():
                token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            if count >= self.min_freq and token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

    def numericalize(self, text: str) -> list[int]:
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(t, unk_id) for t in text.lower().split()]

    def __len__(self):
        return len(self.token_to_id)


class IssueDataset(Dataset):
    """PyTorch Dataset wrapping issue texts and category labels."""

    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary):
        self.labels = labels
        self.encoded_texts = [vocab.numericalize(t) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]


def collate_fn(batch, pad_idx=0, max_len=64):
    """Pads sequences to max length in batch and converts to PyTorch Tensors."""
    sequences, labels = zip(*batch)
    batch_size = len(sequences)

    # Pad sequences
    padded = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


class TextClassifierMLP(nn.Module):
    """
    Neural Architecture:
    Input IDs (B, L) -> Embedding (B, L, D) -> Global Mean Pool (B, D)
    -> Linear (D, H) -> ReLU -> Dropout -> Linear (H, K) -> Logits
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # Mask out padding tokens (index 0) during mean pooling
        mask = (x != 0).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        sum_embeddings = (embedded * mask).sum(dim=1)  # (batch_size, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1e-9)  # (batch_size, 1)
        pooled = sum_embeddings / lengths  # (batch_size, embed_dim)

        # Dense classification head
        hidden = self.relu(self.fc1(pooled))  # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)  # (batch_size, num_classes)
        return logits


def train_pytorch_model():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load data
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    cat_to_id = metadata["category_to_id"]
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    num_classes = len(cat_to_id)

    # 3. Build Vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(train_df["text"].tolist())
    print(f"Constructed Vocabulary Size: {len(vocab)} tokens")

    # Save vocabulary
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(PYTORCH_VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab.token_to_id, f, indent=2)

    # 4. Prepare DataLoaders
    train_labels = [cat_to_id[c] for c in train_df["category"]]
    val_labels = [cat_to_id[c] for c in val_df["category"]]

    train_dataset = IssueDataset(train_df["text"].tolist(), train_labels, vocab)
    val_dataset = IssueDataset(val_df["text"].tolist(), val_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 5. Initialize Model, Loss, Optimizer
    model = TextClassifierMLP(
        vocab_size=len(vocab), embed_dim=64, hidden_dim=32, num_classes=num_classes, pad_idx=0
    ).to(device)

    # Calculate class weights for loss function to handle class imbalance
    class_counts = train_df["category"].value_counts()
    total_samples = len(train_df)
    weights = [total_samples / (num_classes * class_counts[id_to_cat[i]]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # 6. Training Loop
    epochs = 12
    best_val_f1 = 0.0

    print("\n--- Training PyTorch Classifier ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()  # 1. Reset gradients
            outputs = model(texts)  # 2. Forward pass
            loss = criterion(outputs, labels)  # 3. Calculate loss
            loss.backward()  # 4. Backward pass (Autograd)
            optimizer.step()  # 5. Update weights

            total_loss += loss.item() * texts.size(0)

        epoch_loss = total_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro")
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocab),
                    "embed_dim": 64,
                    "hidden_dim": 32,
                    "num_classes": num_classes,
                    "macro_f1": best_val_f1,
                },
                PYTORCH_MODEL_FILE,
            )

    print(f"\nModel training complete. Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"Saved checkpoint to: {PYTORCH_MODEL_FILE}")


if __name__ == "__main__":
    train_pytorch_model()
