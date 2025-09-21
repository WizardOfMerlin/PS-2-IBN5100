import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm

class QueryCategoryDataset(Dataset):
    def __init__(self, queries, categories, labels, encoder):
        self.queries = queries
        self.categories = categories
        self.labels = labels
        self.encoder = encoder
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = str(self.queries[idx])
        category = str(self.categories[idx]).replace('>', ' ')  # Convert path to text
        label = int(self.labels[idx])
        
        # Encode query and category
        query_embedding = self.encoder.encode(query)
        category_embedding = self.encoder.encode(category)
        
        return {
            'query_embedding': torch.tensor(query_embedding, dtype=torch.float32),
            'category_embedding': torch.tensor(category_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class RelevanceClassifier(nn.Module):
    def __init__(self, embedding_dim=384):
        super(RelevanceClassifier, self).__init__()
        # Combine query and category embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, query_emb, category_emb):
        # Concatenate embeddings
        combined = torch.cat([query_emb, category_emb], dim=1)
        return self.classifier(combined)

def load_and_prepare_data(csv_file):
    """Load translated CSV and prepare for training"""
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # Extract relevant columns
    queries = df['origin_query'].tolist()
    categories = df['category_path'].tolist()
    labels = df['label'].tolist()
    
    print(f"Loaded {len(queries)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return queries, categories, labels

def train_model(csv_file, epochs=10, batch_size=64, test_size=0.2):
    """Main training function"""
    
    # Load data
    queries, categories, labels = load_and_prepare_data(csv_file)
    
    # Initialize sentence transformer
    print("Loading sentence transformer...")
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Split data
    print("Splitting data...")
    X_train_q, X_test_q, X_train_c, X_test_c, y_train, y_test = train_test_split(
        queries, categories, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train_q)}")
    print(f"Test samples: {len(X_test_q)}")
    
    # Create datasets
    train_dataset = QueryCategoryDataset(X_train_q, X_train_c, y_train, encoder)
    test_dataset = QueryCategoryDataset(X_test_q, X_test_c, y_test, encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RelevanceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            query_emb = batch['query_embedding'].to(device)
            category_emb = batch['category_embedding'].to(device)
            labels_batch = batch['label'].to(device)
            
            # Forward pass
            outputs = model(query_emb, category_emb)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            query_emb = batch['query_embedding'].to(device)
            category_emb = batch['category_embedding'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(query_emb, category_emb)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions)
    print(f"\nF1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder': encoder,
        'f1_score': f1
    }, 'query_category_model.pt')
    
    print("Model saved as 'query_category_model.pt'")
    return model, encoder, f1

if __name__ == "__main__":
    # Train the model
    csv_file = "translated_training_data_5k.csv"  # Your new 5k translated data
    model, encoder, f1_score = train_model(csv_file)
    print(f"Training complete! F1-Score: {f1_score:.4f}")