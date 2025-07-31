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
CHECKPOINT_DIR = "checkpoints_LSTM"
LOG_FILE = "training_log_LSTM.csv"
BATCH_SIZE = 64
EPOCHS = 50
EMBED_DIM = 64
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
SEED = 42

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# ========= Seed fix ========= #
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ========= Load top domains ========= #
def load_lexical_domains(filepath):
    try:
        with open(filepath, "r") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using empty domains list.")
        return []

def build_domain_length_table(domains):
    """Tạo bảng tra cứu tên miền theo độ dài để tối ưu hóa tìm kiếm"""
    length_table = {}
    for domain in domains:
        domain_len = len(domain)
        if domain_len not in length_table:
            length_table[domain_len] = []
        length_table[domain_len].append(domain)
    return length_table

legit_domains = load_lexical_domains("alexa-top-1000.txt")
# Tạo bảng tra cứu độ dài tên miền để tối ưu hóa
domain_length_table = build_domain_length_table(legit_domains)

# ========= Feature extraction ========= #
def get_entropy(s):
    if not s: return 0
    prob = [n_x / len(s) for x, n_x in Counter(s).items()]
    return -sum(p * log2(p) for p in prob)

def min_levenshtein_distance(domain, domain_length_table):
    """
    Tính khoảng cách Levenshtein tối thiểu với các tên miền hợp lệ.
    Chỉ so sánh với các tên miền có độ dài hơn/kém tối đa 2 ký tự.
    """
    if not domain or not domain_length_table:
        return 1
    
    try:
        domain_len = len(domain)
        max_ratio = 0
        
        # Chỉ so sánh với các tên miền có độ dài trong khoảng [domain_len-2, domain_len+2]
        for check_len in range(max(1, domain_len - 2), domain_len + 3):
            if check_len in domain_length_table:
                for legit_domain in domain_length_table[check_len]:
                    ratio = difflib.SequenceMatcher(None, domain, legit_domain).ratio()
                    if ratio > max_ratio:
                        max_ratio = ratio
                    if ratio > 0.9:  # Early stopping nếu tìm thấy match tốt
                        return 1 - max_ratio
        
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
            'spelling_error': min_levenshtein_distance(domain, domain_length_table),
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
class URLClassifierLSTM(nn.Module):
    def __init__(self, vocab_sizes, feat_dim):
        super().__init__()
        # Embedding layers
        self.char_emb = nn.Embedding(vocab_sizes[0], EMBED_DIM)
        self.bi_emb = nn.Embedding(vocab_sizes[1], EMBED_DIM)
        self.tri_emb = nn.Embedding(vocab_sizes[2], EMBED_DIM)
        
        # LSTM layers for sequence modeling
        self.char_lstm = nn.LSTM(EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_LAYERS, 
                                batch_first=True, dropout=0.2, bidirectional=True)
        self.bi_lstm = nn.LSTM(EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_LAYERS, 
                              batch_first=True, dropout=0.2, bidirectional=True)
        self.tri_lstm = nn.LSTM(EMBED_DIM, LSTM_HIDDEN_DIM, LSTM_LAYERS, 
                               batch_first=True, dropout=0.2, bidirectional=True)
        
        # Feature projection
        self.feat_proj = nn.Linear(feat_dim, EMBED_DIM)
        
        # Final classifier
        # LSTM output dimension: LSTM_HIDDEN_DIM * 2 (bidirectional) for each n-gram type
        lstm_output_dim = LSTM_HIDDEN_DIM * 2 * 3  # 3 LSTM outputs (char, bi, tri)
        total_features = lstm_output_dim + EMBED_DIM  # LSTM features + projected features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, c, b, t, f):
        # Get embeddings
        c_emb = self.char_emb(c)  # [batch_size, seq_len, embed_dim]
        b_emb = self.bi_emb(b)    # [batch_size, seq_len, embed_dim]
        t_emb = self.tri_emb(t)   # [batch_size, seq_len, embed_dim]
        
        # Pass through LSTM layers
        c_lstm_out, (c_hidden, _) = self.char_lstm(c_emb)
        b_lstm_out, (b_hidden, _) = self.bi_lstm(b_emb)
        t_lstm_out, (t_hidden, _) = self.tri_lstm(t_emb)
        
        # Use the last hidden state from both directions
        # hidden shape: [num_layers * num_directions, batch, hidden_size]
        c_final = c_hidden[-2:].transpose(0, 1).contiguous().view(c.size(0), -1)  # Concatenate last forward and backward
        b_final = b_hidden[-2:].transpose(0, 1).contiguous().view(b.size(0), -1)
        t_final = t_hidden[-2:].transpose(0, 1).contiguous().view(t.size(0), -1)
        
        # Project features
        f_proj = self.feat_proj(f)
        
        # Concatenate all features
        x = torch.cat([c_final, b_final, t_final, f_proj], dim=1)
        
        return self.classifier(x)

# ========= Main training ========= #
def train_model(train_csv, val_csv, test_csv):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(csv_path):
        feat_csv = csv_path.replace('.csv', '_features_LSTM.csv')
        if os.path.exists(feat_csv):
            print(f"Loading cached features from {feat_csv}")
            df = pd.read_csv(feat_csv)
        else:
            print(f"Processing {csv_path}...")
            df = pd.read_csv(csv_path, low_memory=False)  # Fix dtype warning
            print(f"Loaded {len(df)} rows from {csv_path}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Check if required columns exist
            if 'label' not in df.columns:
                print("Error: 'label' column not found!")
                return pd.DataFrame()
            if 'url' not in df.columns:
                print("Error: 'url' column not found!")
                return pd.DataFrame()
            
            print(f"Label distribution: {df['label'].value_counts()}")
            df['label_enc'] = df['label'].map({'good': 0, 'bad': 1})
            print(f"Rows after label encoding: {len(df)}")
            df = df.dropna()
            print(f"Rows after dropping NaN: {len(df)}")
            
            if len(df) == 0:
                print("Warning: No data remaining after preprocessing!")
                return df
            
            print("🔄 Extracting features...")
            feats = df['url'].apply(extract_features)
            print("🔄 Adding n-gram features...")
            # Add bigram and trigram features
            df['bigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 2)))
            df['trigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 3)))
            print("🔄 Converting features to DataFrame...")
            # Handle different pandas behavior
            if isinstance(feats, pd.DataFrame):
                # pandas automatically expanded Series to DataFrame
                for col in feats.columns:
                    df[col] = feats[col]
            else:
                # feats is a Series of Series, need to convert
                feats_df = pd.DataFrame(feats.tolist(), index=feats.index)
                for col in feats_df.columns:
                    df[col] = feats_df[col]
            df.to_csv(feat_csv, index=False)
            print(f"💾 Saved features to {feat_csv}")
        
        print(f"Final dataset shape: {df.shape}")
        return df

    df_train = prepare_features(train_csv)
    df_val = prepare_features(val_csv)
    df_test = prepare_features(test_csv)

    print(f"\n🔤 Building vocabularies...")
    char_vocab = build_vocab(df_train['url'], 1)
    print(f"   Character vocab size: {len(char_vocab)}")
    bi_vocab = build_vocab(df_train['url'], 2)
    print(f"   Bigram vocab size: {len(bi_vocab)}")
    tri_vocab = build_vocab(df_train['url'], 3)
    print(f"   Trigram vocab size: {len(tri_vocab)}")

    print(f"\n📊 Creating datasets...")
    train_ds = URLDataset(df_train, char_vocab, bi_vocab, tri_vocab)
    val_ds = URLDataset(df_val, char_vocab, bi_vocab, tri_vocab)
    test_ds = URLDataset(df_test, char_vocab, bi_vocab, tri_vocab)

    print(f"📦 Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"🧠 Initializing LSTM model...")
    model = URLClassifierLSTM((len(char_vocab), len(bi_vocab), len(tri_vocab)), feat_dim=train_ds.feats.shape[1])
    model = model.to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Early stopping variables
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_lstm.pt")

    # Load checkpoint if exists and compatible
    latest_ckpt = None
    if os.path.exists(CHECKPOINT_DIR):
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_lstm_epoch") and f.endswith(".pt")]
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

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING - {EPOCHS} epochs")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        model.train()
        total_loss, total_pred, total_true = 0, [], []
        batch_count = 0

        for i, (c, b, t, f, y) in enumerate(train_loader):
            c, b, t, f, y = c.to(device), b.to(device), t.to(device), f.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(c, b, t, f)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * c.size(0)
            total_pred += torch.argmax(logits, 1).tolist()
            total_true += y.tolist()
            batch_count += 1
            
            # Progress indicator every 10 batches
            if (i + 1) % 10 == 0:
                current_loss = total_loss / ((i + 1) * BATCH_SIZE)
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {current_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader.dataset)
        acc = accuracy_score(total_true, total_pred)
        
        print(f"✅ Training completed - Loss: {avg_train_loss:.4f}, Accuracy: {acc:.4f}")

        print(f"✅ Training completed - Loss: {avg_train_loss:.4f}, Accuracy: {acc:.4f}")

        # Validation
        print(f"🔍 Running validation...")
        model.eval()
        val_probs, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for i, (c, b, t, f, y) in enumerate(val_loader):
                c, b, t, f, y = c.to(device), b.to(device), t.to(device), f.to(device), y.to(device)
                logits = model(c, b, t, f)
                prob = torch.softmax(logits, 1)[:, 1]
                val_probs += prob.cpu().tolist()
                val_preds += torch.argmax(logits, 1).cpu().tolist()
                val_labels += y.cpu().tolist()
                
                # Progress for validation
                if (i + 1) % 5 == 0:
                    print(f"  Val batch {i+1}/{len(val_loader)}")
                    
        auc_val = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"\n📊 EPOCH {epoch+1} RESULTS:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {acc:.4f}")
        print(f"   Val Acc: {val_acc:.4f} | Val AUC: {auc_val:.4f}")
        print(classification_report(val_labels, val_preds, target_names=["good", "bad"]))

        # Early stopping logic
        if auc_val > best_val_auc + EARLY_STOPPING_MIN_DELTA:
            best_val_auc = auc_val
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best validation AUC: {best_val_auc:.4f} - Best model saved!")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{EARLY_STOPPING_PATIENCE} epochs")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
                print(f"🏆 Best validation AUC achieved: {best_val_auc:.4f}")
                break

        # Save regular checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_lstm_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_auc", "timestamp"])
            writer.writerow([epoch+1, avg_train_loss, acc, val_acc, auc_val, datetime.now().isoformat()])
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📝 Training log saved to: {LOG_FILE}")
    print(f"💾 Model checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"🏆 Best validation AUC achieved: {best_val_auc:.4f}")
    print(f"💎 Best model saved at: {best_model_path}")

def test_model(train_csv, val_csv, test_csv):
    """Test model with latest checkpoint on test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_features(csv_path):
        feat_csv = csv_path.replace('.csv', '_features_LSTM.csv')
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            df = pd.read_csv(csv_path, low_memory=False)
            df['label_enc'] = df['label'].map({'good': 0, 'bad': 1})
            df = df.dropna()
            feats = df['url'].apply(extract_features)
            df['bigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 2)))
            df['trigram_count'] = df['url'].apply(lambda x: len(get_char_ngrams(x, 3)))
            # Handle different pandas behavior
            if isinstance(feats, pd.DataFrame):
                # pandas automatically expanded Series to DataFrame
                for col in feats.columns:
                    df[col] = feats[col]
            else:
                # feats is a Series of Series, need to convert
                feats_df = pd.DataFrame(feats.tolist(), index=feats.index)
                for col in feats_df.columns:
                    df[col] = feats_df[col]
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

    # Initialize LSTM model
    model = URLClassifierLSTM((len(char_vocab), len(bi_vocab), len(tri_vocab)), feat_dim=test_ds.feats.shape[1])
    model = model.to(device)

    # Load latest checkpoint (prefer best model if available)
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_lstm.pt")
    latest_ckpt = None
    
    if os.path.exists(best_model_path):
        latest_ckpt = best_model_path
        print(f"Found best model: {best_model_path}")
    elif os.path.exists(CHECKPOINT_DIR):
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_lstm_epoch") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
            print(f"Found latest checkpoint: {latest_ckpt}")
    
    if not latest_ckpt:
        print("No checkpoint found! Please train the model first.")
        return
    
    try:
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint)
        
        if "best_model" in latest_ckpt:
            print(f"Successfully loaded best model")
            epoch_num = "best"
        else:
            epoch_num = int(re.findall(r'\d+', os.path.basename(latest_ckpt))[-1])
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
    print(f"TEST RESULTS (Model: {epoch_num})")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["good", "bad"]))
    
    # Save test results
    test_results_file = "test_results_LSTM.csv"
    with open(test_results_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_acc", "test_auc", "timestamp"])
        writer.writerow([epoch_num, test_acc, test_auc, datetime.now().isoformat()])
    
    print(f"\nTest results saved to: {test_results_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Training: python train_phishing_detector_LSTM.py train train.csv val.csv test.csv")
        print("  Testing:  python train_phishing_detector_LSTM.py test train.csv val.csv test.csv")
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode == "train":
        if len(sys.argv) != 5:
            print("Usage: python train_phishing_detector_LSTM.py train train.csv val.csv test.csv")
            sys.exit(1)
        train_model(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "test":
        if len(sys.argv) != 5:
            print("Usage: python train_phishing_detector_LSTM.py test train.csv val.csv test.csv")
            sys.exit(1)
        test_model(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Invalid mode. Use 'train' or 'test'")
        sys.exit(1)
