import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class QueryCategoryPredictor:
    def __init__(self, model_path='query_category_model.pt'):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.encoder = checkpoint['encoder']
        
        self.model = RelevanceClassifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded! F1-Score: {checkpoint['f1_score']:.4f}")
    
    def predict(self, query, category_path):
        query = str(query)
        category = str(category_path).replace('>', ' ')
        
        query_emb = torch.tensor(self.encoder.encode(query), dtype=torch.float32).unsqueeze(0)
        category_emb = torch.tensor(self.encoder.encode(category), dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            query_emb = query_emb.to(self.device)
            category_emb = category_emb.to(self.device)
            outputs = self.model(query_emb, category_emb)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    
    def predict_batch(self, queries, category_paths):
        """Predict relevance for multiple query-category pairs"""
        predictions = []
        confidences = []
        
        for query, category in zip(queries, category_paths):
            pred, conf = self.predict(query, category)
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences

# Import the model class
from train_ai_model import RelevanceClassifier

def test_model():
    """Test the trained model with some examples"""
    predictor = QueryCategoryPredictor()
    
    # Test examples
    test_cases = [
        ("red heels", "Shoes>Women>Heels"),
        ("laptop", "Electronics>Computers>Laptops"),
        ("running shoes", "Sports>Running>Footwear"),
        ("coffee", "Electronics>Audio>Speakers"),  # Should be 0
        ("smartphone", "Books>Fiction>Romance")     # Should be 0
    ]
    
    print("Testing model predictions:")
    print("-" * 50)
    
    for query, category in test_cases:
        prediction, confidence = predictor.predict(query, category)
        relevance = "Relevant" if prediction == 1 else "Not Relevant"
        print(f"Query: '{query}'")
        print(f"Category: '{category}'")
        print(f"Prediction: {relevance} (Confidence: {confidence:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    test_model()