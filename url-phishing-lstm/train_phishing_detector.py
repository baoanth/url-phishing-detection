import os
import re
import sys
import csv
import torch
import random
import difflib
import pandas as pd
import numpy as np
from collections import Counter
from urllib.parse import urlparse
from math import log2
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ========= Settings ========= #
MAX_LEN = 200
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.csv"
BATCH_SIZE = 64
EPOCHS = 50
EMBED_DIM = 64
SEED = 42

# ========= Seed fix ========= #
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ========= Load top domains ========= #
def load_lexical_domains(filepath):
    with open(filepath, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]
legit_domains = load_lexical_domains("alexa-top-1000.txt")

# ========= Feature extraction ========= #
def get_entropy(s):
    if not s: return 0
    prob = [n_x / len(s) for x, n_x in Counter(s).items()]
    return -sum(p * log2(p) for p in prob)

def min_levenshtein_distance(domain, legit_domains):
    try:
        # Tối ưu: chỉ so sánh với top 100 domains và có early stopping
        max_ratio = 0
        for legit in legit_domains[:100]:  # Giới hạn chỉ 100 domain đầu
            ratio = difflib.SequenceMatcher(None, domain, legit).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
            if ratio > 0.9:  # Early stopping nếu tìm thấy match tốt
                break
        return 1 - max_ratio
    except:
        return 1

def extract_features(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        features = {
            'url_length': len(url),
            'hostname_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'count_dots': url.count('.'),
            'count_hyphens': url.count('-'),
            'count_at': url.count('@'),
            'count_question': url.count('?'),
            'count_equal': url.count('='),
            'count_http': url.count('http'),
            'count_https': url.count('https'),
            'count_www': url.count('www'),
            'count_digits': sum(c.isdigit() for c in url),
            'count_slash': url.count('/'),
            'count_percent': url.count('%'),
            'count_colon': url.count(':'),
            'is_https': 1 if parsed.scheme == 'https' else 0,
            'count_subdomains': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0,
            'has_ip': 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0,
            'has_suspicious_words': 1 if any(word in url.lower() for word in ['login', 'secure', 'update', 'verify', 'bank', 'account']) else 0,
            'spelling_error': min_levenshtein_distance(domain, legit_domains),
            'url_entropy': get_entropy(url),
            'domain_entropy': get_entropy(domain)
        }
    except:
        features = {k:0 for k in [
            'url_length', 'hostname_length', 'path_length', 'count_dots', 'count_hyphens', 'count_at',
            'count_question', 'count_equal', 'count_http', 'count_https', 'count_www', 'count_digits',
            'count_slash', 'count_percent', 'count_colon', 'is_https', 'count_subdomains',
            'has_ip', 'has_suspicious_words', 'spelling_error', 'url_entropy', 'domain_entropy'
        ]}
    print(url)
    return pd.Series(features)

# ========= Tokenizer ========= #
def get_char_ngrams(s, n):
    return [s[i:i+n] for i in range(len(s)-n+1)]

def build_vocab(urls, ngram=1):
    vocab = set()
    for url in urls:
        tokens = get_char_ngrams(url.lower(), ngram)
        vocab.update(tokens)
    vocab = {tok: idx+1 for idx, tok in enumerate(sorted(vocab))}
    vocab['<PAD>'] = 0
    return vocab

def tokenize_url(url, vocab, ngram=1, max_len=MAX_LEN):
    tokens = get_char_ngrams(url.lower(), ngram)
    ids = [vocab.get(tok, 0) for tok in tokens]
    return ids[:max_len] + [0] * (max_len - len(ids[:max_len]))

# ========= Dataset ========= #
class URLDataset(Dataset):
    def __init__(self, df, char_vocab, bi_vocab, tri_vocab):
        self.urls = df['url'].tolist()
        self.labels = df['label_enc'].values
        self.char_vocab = char_vocab
        self.bi_vocab = bi_vocab
        self.tri_vocab = tri_vocab
        self.feats = df[[c for c in df.columns if c not in ['url', 'label', 'label_enc']]].values.astype(np.float32)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        return (
            torch.tensor(tokenize_url(url, self.char_vocab, 1), dtype=torch.long),
            torch.tensor(tokenize_url(url, self.bi_vocab, 2), dtype=torch.long),
            torch.tensor(tokenize_url(url, self.tri_vocab, 3), dtype=torch.long),
            torch.tensor(self.feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ========= Model ========= #
class URLClassifier(nn.Module):
    def __init__(self, vocab_sizes, feat_dim):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_sizes[0], EMBED_DIM)
        self.bi_emb = nn.Embedding(vocab_sizes[1], EMBED_DIM)
        self.tri_emb = nn.Embedding(vocab_sizes[2], EMBED_DIM)
        self.feat_proj = nn.Linear(feat_dim, EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM*4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, c, b, t, f):
        c = self.char_emb(c).mean(dim=1)
        b = self.bi_emb(b).mean(dim=1)
        t = self.tri_emb(t).mean(dim=1)
        f = self.feat_proj(f)
        x = torch.cat([c, b, t, f], dim=1)
        return self.classifier(x)

# ========= Main training ========= #
def train_model(train_csv, val_csv, test_csv):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(csv_path):
        feat_csv = csv_path.replace('.csv', '_features.csv')
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            df = pd.read_csv(csv_path)
            df['label_enc'] = df['label'].map({'good': 0, 'bad': 1})
            df = df.dropna()
            feats = df['url'].apply(extract_features)
            # Add bigram and trigram features
            df['bigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 2)))
            df['trigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 3)))
            df[feats.columns] = feats
            df.to_csv(feat_csv, index=False)
        return df

    df_train = prepare_features(train_csv)
    df_val = prepare_features(val_csv)
    df_test = prepare_features(test_csv)

    char_vocab = build_vocab(df_train['url'], 1)
    bi_vocab = build_vocab(df_train['url'], 2)
    tri_vocab = build_vocab(df_train['url'], 3)

    train_ds = URLDataset(df_train, char_vocab, bi_vocab, tri_vocab)
    val_ds = URLDataset(df_val, char_vocab, bi_vocab, tri_vocab)
    test_ds = URLDataset(df_test, char_vocab, bi_vocab, tri_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = URLClassifier((len(char_vocab), len(bi_vocab), len(tri_vocab)), feat_dim=train_ds.feats.shape[1])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if exists and compatible
    latest_ckpt = None
    if os.path.exists(CHECKPOINT_DIR):
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
    
    start_epoch = 0
    if latest_ckpt:
        try:
            print(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint)
            start_epoch = int(re.findall(r'\d+', ckpts[-1])[-1])
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Cannot load checkpoint due to model structure mismatch: {e}")
            print("Starting training from scratch...")
            start_epoch = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss, total_pred, total_true = 0, [], []

        for c, b, t, f, y in train_loader:
            c, b, t, f, y = c.to(device), b.to(device), t.to(device), f.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(c, b, t, f)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * c.size(0)
            total_pred += torch.argmax(logits, 1).tolist()
            total_true += y.tolist()

        avg_train_loss = total_loss / len(train_loader.dataset)
        acc = accuracy_score(total_true, total_pred)

        # Validation
        model.eval()
        val_probs, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for c, b, t, f, y in val_loader:
                c, b, t, f, y = c.to(device), b.to(device), t.to(device), f.to(device), y.to(device)
                logits = model(c, b, t, f)
                prob = torch.softmax(logits, 1)[:, 1]
                val_probs += prob.cpu().tolist()
                val_preds += torch.argmax(logits, 1).cpu().tolist()
                val_labels += y.cpu().tolist()
        auc_val = roc_auc_score(val_labels, val_probs)

        print(f"\n[Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | Acc: {acc:.4f} | Val AUC: {auc_val:.4f}")
        print(classification_report(val_labels, val_preds, target_names=["good", "bad"]))

        # Save
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "train_acc", "val_auc", "timestamp"])
            writer.writerow([epoch+1, avg_train_loss, acc, auc_val, datetime.now().isoformat()])

def test_model(train_csv, val_csv, test_csv):
    """Test model with latest checkpoint on test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_features(csv_path):
        feat_csv = csv_path.replace('.csv', '_features.csv')
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            df = pd.read_csv(csv_path)
            df['label_enc'] = df['label'].map({'good': 0, 'bad': 1})
            df = df.dropna()
            feats = df['url'].apply(extract_features)
            df['bigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 2)))
            df['trigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 3)))
            df[feats.columns] = feats
            df.to_csv(feat_csv, index=False)
        return df

    # Load datasets
    df_train = prepare_features(train_csv)
    df_val = prepare_features(val_csv)
    df_test = prepare_features(test_csv)

    # Build vocabularies from training data
    char_vocab = build_vocab(df_train['url'], 1)
    bi_vocab = build_vocab(df_train['url'], 2)
    tri_vocab = build_vocab(df_train['url'], 3)

    # Create test dataset
    test_ds = URLDataset(df_test, char_vocab, bi_vocab, tri_vocab)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize model
    model = URLClassifier((len(char_vocab), len(bi_vocab), len(tri_vocab)), feat_dim=test_ds.feats.shape[1])
    model = model.to(device)

    # Load latest checkpoint
    latest_ckpt = None
    if os.path.exists(CHECKPOINT_DIR):
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
    
    if not latest_ckpt:
        print("No checkpoint found! Please train the model first.")
        return
    
    try:
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint)
        epoch_num = int(re.findall(r'\d+', ckpts[-1])[-1])
        print(f"Successfully loaded checkpoint from epoch {epoch_num}")
    except RuntimeError as e:
        print(f"Cannot load checkpoint: {e}")
        return

    # Test evaluation
    model.eval()
    test_probs, test_preds, test_labels = [], [], []
    
    print("Running test evaluation...")
    with torch.no_grad():
        for c, b, t, f, y in test_loader:
            c, b, t, f, y = c.to(device), b.to(device), t.to(device), f.to(device), y.to(device)
            logits = model(c, b, t, f)
            prob = torch.softmax(logits, 1)[:, 1]
            test_probs += prob.cpu().tolist()
            test_preds += torch.argmax(logits, 1).cpu().tolist()
            test_labels += y.cpu().tolist()
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS (Epoch {epoch_num})")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["good", "bad"]))
    
    # Save test results
    test_results_file = "test_results.csv"
    with open(test_results_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "test_acc", "test_auc", "timestamp"])
        writer.writerow([epoch_num, test_acc, test_auc, datetime.now().isoformat()])
    
    print(f"\nTest results saved to: {test_results_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Training: python train_phishing_detector.py train train.csv val.csv test.csv")
        print("  Testing:  python train_phishing_detector.py test train.csv val.csv test.csv")
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode == "train":
        if len(sys.argv) != 5:
            print("Usage: python train_phishing_detector.py train train.csv val.csv test.csv")
            sys.exit(1)
        train_model(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "test":
        if len(sys.argv) != 5:
            print("Usage: python train_phishing_detector.py test train.csv val.csv test.csv")
            sys.exit(1)
        test_model(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Invalid mode. Use 'train' or 'test'")
        sys.exit(1)
