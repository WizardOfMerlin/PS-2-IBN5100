Multilingual Query-Category Relevance System

Problem Statement
This system solves the Multilingual Query-Category Relevance task for e-commerce platforms. It determines whether a user's search query is semantically relevant to a given product category hierarchy.

Features

Core Capabilities
- Multilingual Support: Handles 20+ languages including English, Spanish, French, German, Chinese, Japanese, Korean, Russian, and more
- Advanced Rule-Based System: Sophisticated heuristics with domain-specific knowledge
- Real-time Inference: Fast batch processing with progress tracking
- Confidence Scoring: Each prediction includes confidence levels
- Interactive Dashboard: Beautiful Streamlit interface with data visualization

Algorithm Features
- Keyword Matching: Direct and partial word overlap analysis
- Category-Specific Rules: Domain knowledge for electronics, clothing, sports, beauty, etc.
- Brand Recognition: Identifies and matches common brand names
- Language Detection: Automatic language identification and handling
- Query Analysis: Length-based heuristics and sentiment detection


Installation & Setup

Quick Start
1. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Application**
   ```bash
   python3 -m streamlit run query_category_relevance_app.py
   ```

3. Access the Web Interface**
   - Open your browser and go to: `http://localhost:8501`

Usage Guide

1. Data Format
Your CSV file must contain these columns:
- Query: The search query (e.g., "red running shoes")
- L1: Top-level category (e.g., "Sports & Outdoors")
- L2: Mid-level category (e.g., "Athletic Shoes")
- L3: Leaf category (e.g., "Running Shoes")

2. Upload & Process
1. 
2. Review the dataset overview and language distribution
3. Click "Run Inference" to generate predictions
4. Download results with predictions and confidence scores

3. Output Format
The system generates:
- Prediction: 1 (Relevant) or 0 (Not Relevant)
- Confidence: Numerical confidence score (0.0 to 1.0)
- Prediction_Label: Human-readable label

Model Performance

Evaluation Metric
- Primary: F1-Score on positive class (Relevant = 1)
- Formula: F1 = (2 × Precision × Recall) / (Precision + Recall)

System Strengths
- High Precision: Advanced heuristics minimize false positives
- Language Adaptability: Unicode support and cross-lingual patterns
- Domain Knowledge: Category-specific rules for better accuracy
- Scalability: Efficient batch processing for large datasets

Advanced Features

Language Detection
Automatic identification of:
- Romance Languages: Spanish, French, Italian, Portuguese
- Germanic Languages: German, Dutch, English
- Slavic Languages: Russian, Polish, Czech
- Asian Languages: Chinese, Japanese, Korean
- And more: 20+ languages supported

Category Intelligence
Domain-specific knowledge for:
- Electronics: Phones, laptops, cameras, audio devices
- Clothing: Shirts, shoes, accessories, seasonal wear
- Sports: Equipment, fitness items, outdoor gear
- Beauty: Makeup, skincare, fragrances
- Home & Kitchen: Appliances, furniture, decor
- Automotive: Parts, accessories, maintenance items

Smart Matching
- Exact Word Matching: Direct overlap scoring
- Partial Matching: Substring and similarity detection
- Brand Recognition: Common brand name identification
- Negative Sentiment: Detection of exclusion terms
- Query Complexity: Length and specificity analysis

Performance Optimization

Batch Processing
- Configurable batch sizes for memory optimization
- Progress tracking with visual indicators
- Efficient tensor operations

Technical Architecture

Core Components
1. Data Preprocessor: Text cleaning and normalization
2. Language Detector: Multilingual text analysis
3. Feature Extractor: Query and category feature engineering
4. Rule Engine: Advanced heuristic scoring system
5. Confidence Calculator: Prediction reliability assessment   (removed from front end but still exists)
6. Results Manager: Output formatting and export
