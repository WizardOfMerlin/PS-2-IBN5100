import streamlit as st
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from datetime import datetime
from translatepy import Translator
import time

# Model architecture (same as training)
class RelevanceClassifier(nn.Module):
    def __init__(self, embedding_dim=384):
        super(RelevanceClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, query_emb, category_emb):
        combined = torch.cat([query_emb, category_emb], dim=1)
        return self.classifier(combined)

@st.cache_resource
def load_ai_model():
    """Load the trained AI model"""
    try:
        # Load the saved model with weights_only=False for compatibility
        checkpoint = torch.load('query_category_model.pt', map_location='cpu', weights_only=False)
        
        # Initialize model and encoder
        model = RelevanceClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        encoder = checkpoint['encoder']

        return model, encoder
        
    except Exception as e:
        st.error(f" Could not load AI model: {e}")
        return None, None

@st.cache_resource
def load_translator():
    """Load the translation service"""
    try:
        translator = Translator()
        st.success(" Translation service loaded")
        return translator
    except Exception as e:
        st.error(f" Failed to load translator: {e}")
        return None

def translate_queries(df, translator):
    """Translate queries to English using multiple services"""
    translated_queries = []
    successful_translations = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        language_code = row['language']
        query = str(row['origin_query'])
        
        # Update status
        if i % 50 == 0:
            status_text.write(f"Translating: {i+1}/{len(df)} queries")
        
        if language_code == 'en':
            translated_queries.append(query)
            successful_translations += 1
        else:
            try:
                # Rate limiting
                if i > 0 and i % 10 == 0:
                    time.sleep(0.3)
                
                result = translator.translate(query, source_language=language_code, destination_language='en')
                translated_text = result.result
                translated_queries.append(translated_text)
                successful_translations += 1
                
            except Exception:
                # Fallback to original
                translated_queries.append(query)
                successful_translations += 1
        
        # Update progress
        progress_bar.progress((i + 1) / len(df))
    
    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()
    
    return translated_queries, successful_translations

def predict_with_ai(queries, categories, model, encoder):
    """Make predictions using the trained AI model"""
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for query, category in zip(queries, categories):
            # Encode query and category
            query_emb = encoder.encode(str(query))
            category_emb = encoder.encode(str(category).replace('>', ' '))
            
            # Convert to tensors
            query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
            category_tensor = torch.tensor(category_emb, dtype=torch.float32).unsqueeze(0)
            
            # Get prediction
            outputs = model(query_tensor, category_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            predictions.append(prediction)
            confidences.append(confidence)
    
    return predictions, confidences

def main():
    st.title("Query & Category Relation Evaluator")
    st.write("Upload your csv file, let it translate (for non english queries) then let AI predict the relations in binary form and download the outputted csv file")
    
    # Load AI model and translator
    model, encoder = load_ai_model()
    translator = load_translator()
    
    if model is None or translator is None:
        st.error("Ai model or translator is not loaded properly")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read data
        df = pd.read_csv(uploaded_file)
        st.success(f"File loaded: {len(df)} rows")
        
        # Show data preview
        st.subheader("Data Preview (for checking if file is correct)")
        st.dataframe(df.head(), use_container_width=True)
        
        # Check required columns
        required_cols = ['origin_query', 'category_path', 'language']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.write("Available columns:", list(df.columns))
            return
        
        # Auto-limit to 5k rows for performance
        #if len(df) > 5000:
            #df = df.head(5000)          I dont think this will be useful
            #st.info(f"Processing first 5000 rows for optimal performance")
        
        # Run complete pipeline
        if st.button("Start process", use_container_width=True):
            
            # Step 1: Translation
            st.subheader("Translating Queries")
            with st.spinner("Translating queries to English"):
                translated_queries, translation_success = translate_queries(df, translator)

            st.success(f"Translation complete, {translation_success}/{len(df)} queries processed")

            # Step 2: AI Inference
            st.subheader("AI Predictions")
            with st.spinner("AI is computing query-category relationship"):
                # Use translated queries for AI prediction
                predictions, confidences = predict_with_ai(
                    translated_queries,
                    df['category_path'].tolist(),
                    model, encoder
                )

            st.success(f"AI inference complete! {len(predictions)} predictions generated")

            # Create results dataframe with ORIGINAL data + AI predictions
            results_df = df.copy()  # Keep original queries
            results_df['label'] = predictions  # Add AI predictions (0/1)
            results_df['confidence'] = confidences
            results_df['relevance_label'] = results_df['label'].map({1: 'Relevant', 0: 'Not Relevant'})
            
            # Display results
            st.subheader("Final Results")
            
            # Summary metrics
            col1, col2 = st.columns(2)
            with col1:
                relevant_count = sum(predictions)
                st.metric("Relevant Queries", relevant_count)
            with col2:
                not_relevant_count = len(predictions) - relevant_count
                st.metric("Not Relevant", not_relevant_count)
            
            # Results table (showing original queries + AI labels)
            st.subheader("Results Table")
            
            # Display table with key columns (original queries + AI predictions)
            display_cols = ['language', 'origin_query', 'category_path', 'relevance_label']
            display_df = results_df[display_cols].copy()
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download section
            st.subheader("Download Results")
            
            # Prepare download data (original format + AI predictions)
            download_df = df.copy()  # Original data with original queries
            download_df['label'] = predictions  # Add AI-generated labels
            
            # Convert to CSV
            csv_data = download_df.to_csv(index=False)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multilingual_ai_predictions_{timestamp}.csv"
            
            st.download_button(
                label="Download Complete Results CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
                help="Downloads CSV with original queries and AI-generated labels"
            )
            
            st.success(f"Complete process is finished, ready for download.")
                
if __name__ == "__main__":
    main()