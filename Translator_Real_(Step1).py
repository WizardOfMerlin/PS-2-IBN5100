import streamlit as st
import pandas as pd
from translatepy import Translator
import time

st.title("🌍 Multi-Service Query Translator")
st.write("Upload a CSV file and translate queries using multiple translation services")

@st.cache_resource
def load_translator():
    try:
        translator = Translator()
        return translator
    except Exception as e:
        st.error(f"Failed to load translator: {e}")
        return None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"File uploaded successfully! Rows,Columns : {df.shape}")
        
        st.subheader("First 5 rows:")
        st.write("To check if the file is read correctly")
        st.dataframe(df.head())
        
        translator = load_translator()
        
        if 'origin_query' in df.columns and 'language' in df.columns and translator:
            
            # Show language distribution
            lang_counts = df['language'].value_counts()
            st.subheader("Language Distribution:")
            st.write(lang_counts)
            
            # Limit to 5k rows option
            if len(df) > 5000:
                use_subset = st.checkbox(f"Process only first 5000 rows (recommended for speed)")
                if use_subset:
                    df = df.head(5000)
                    st.info(f"Processing first 5000 rows only")
            
            if st.button("🚀 Translate Queries to English"):
                st.write("Starting translation with multiple services...")
                
                # Calculate estimated time
                non_english_count = len(df[df['language'] != 'en'])
                estimated_minutes = (non_english_count * 0.5) / 60  # 0.5 seconds per query
                st.write(f"Estimated time: {estimated_minutes:.1f} minutes for {non_english_count} non-English queries")
                
                progress_bar = st.progress(0)
                translated_queries = []
                successful_translations = 0
                failed_translations = 0
                
                for i, row in df.iterrows():
                    language_code = row['language']
                    query = str(row['origin_query'])
                    
                    # Skip translation for English queries
                    if language_code == 'en':
                        translated_queries.append(query)
                        successful_translations += 1
                    else:
                        try:
                            # Add small delay to avoid rate limiting
                            if i > 0 and i % 10 == 0:
                                time.sleep(0.5)
                            
                            result = translator.translate(query, source_language=language_code, destination_language='en')
                            translated_text = result.result
                            translated_queries.append(translated_text)
                            successful_translations += 1
                            
                        except Exception as e:
                            # Fallback to original query with language marker
                            translated_queries.append(f"[{language_code}] {query}")
                            failed_translations += 1
                    
                    # Update progress every 10 rows
                    if i % 10 == 0 or i == len(df) - 1:
                        progress_bar.progress((i + 1) / len(df))
                
                # Add translated column (use origin_query for consistency with training)
                df['query_english'] = translated_queries
                
                # For AI training, replace origin_query with English translations
                df_for_training = df.copy()
                df_for_training['origin_query'] = translated_queries
                
                st.success("🎉 Translation complete!")
                
                # Auto-save training file
                training_filename = "translated_training_data_5k.csv"
                df_for_training.to_csv(training_filename, index=False)
                st.success(f"💾 **Training data auto-saved as: `{training_filename}`**")
                st.info("🚀 Ready for AI training! Run: `python train_ai_model.py`")
                
                # Statistics
                st.subheader("Translation Statistics:")
                st.write(f"- **Total queries:** {len(df)}")
                st.write(f"- **Successfully translated:** {successful_translations}")
                st.write(f"- **Failed translations:** {failed_translations}")
                st.write(f"- **Success rate:** {successful_translations/len(df)*100:.1f}%")
                
                # Show translated data
                st.subheader("Translation Results:")
                comparison_df = df[['language', 'origin_query', 'query_english']].head(20)
                st.dataframe(comparison_df)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download with both original and translated columns (for analysis)
                    csv_analysis = df.to_csv(index=False)
                    st.download_button(
                        label="� Download Analysis CSV",
                        data=csv_analysis,
                        file_name="translated_analysis.csv",
                        mime="text/csv",
                        help="Contains both original and translated queries for comparison"
                    )
                
                with col2:
                    # Download ready for AI training (origin_query = English translations)
                    csv_training = df_for_training.to_csv(index=False)
                    st.download_button(
                        label="🤖 Download Training CSV",
                        data=csv_training,
                        file_name="translated_training_data_5k.csv",
                        mime="text/csv",
                        help="Ready for AI model training - origin_query contains English translations"
                    )
            
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")