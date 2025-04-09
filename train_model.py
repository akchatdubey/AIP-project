import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import urllib.request

# Define common stopwords (most frequent English words that don't add meaning)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
])

print("Downloading fake news dataset...")

# Use a better fake news dataset (Kaggle dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
# We'll use a simplified version that's publicly available
fake_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/buzzfeed/buzzfeed_fake_news_content.csv"
real_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/buzzfeed/buzzfeed_real_news_content.csv"

# Create a data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the datasets
fake_path = "data/fake_news.csv"
real_path = "data/real_news.csv"

try:
    print("Downloading fake news data...")
    urllib.request.urlretrieve(fake_url, fake_path)
    print("Downloading real news data...")
    urllib.request.urlretrieve(real_url, real_path)
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Using sample data instead...")
    
    # Create sample data if download fails
    # This is a small sample for testing with more clearly fake news examples
    fake_texts = [
        "BREAKING: Obama Signs Executive Order Banning The Pledge Of Allegiance In Schools Nationwide",
        "Pope Francis Shocks World Endorses Donald Trump for President",
        "FBI Agent Suspected In Hillary Email Leaks Found Dead In Apparent Murder-Suicide",
        "BREAKING: Proof That Obama Was Born In Kenya - Trump Was Right All Along",
        "NASA Confirms Earth Will Experience 15 Days Of Complete Darkness In November 2017",
        "BREAKING: Hillary Clinton Filed For Divorce In New York Courts",
        "BREAKING: Trump To DISMANTLE Lady Liberty And RETURN It To France",
        "BREAKING: Trump Offering Free One-Way Tickets to Africa & Mexico for Those Who Wanna Leave America",
        "BREAKING: CIA Agent Confesses On Deathbed: We Blew Up WTC7 On 9/11",
        "BREAKING: Eminem Protesting Trump By BURNING American Flag At Concert", 
        "BREAKING: Congress Announces Barack Obama Indicted For Illegal Wiretapping",
        "Scientists Discover Cancer Is Caused By Government Chemtrail Experiments",
        "Secret Document Reveals Queen Elizabeth Is Actually An Alien Lizard Person",
        "Bill Gates Admits Vaccines Are Population Control Mechanism",
        "BREAKING: New Evidence Shows The Moon Landing Was Faked By Hollywood",
        "SHOCKING: Doctors Find Microchips in COVID-19 Vaccines That Track Your Every Move",
        "ALERT: 5G Towers Confirmed to Cause Coronavirus Infections in Laboratory Tests",
        "URGENT: United Nations Passing Law to Establish World Government by Next Month",
        "BOMBSHELL: Former CIA Director Admits Aliens Control World Governments",
        "BREAKING: Famous Celebrity Rises From Dead After Three Days in Tomb",
        "ALERT: Scientists Find Proof That Drinking Water Is Being Poisoned to Control Population",
        "BREAKING: Government Admits Weather Control Program for Military Purposes",
        "SHOCKING: Secret Document Shows NASA Has Been Hiding Evidence of Ancient Civilization on Mars",
        "BOMBSHELL: World Leaders Caught on Camera Transforming Into Reptilian Forms",
        "BREAKING: Doctor Who Discovered Cure for All Cancers Found Dead Under Mysterious Circumstances"
    ]
    
    real_texts = [
        "Senate Republicans unveil new healthcare bill",
        "US job growth accelerates in June wages continue to lag",
        "Trump Asian allies seek counter to North Korean menace",
        "Economic growth in US leaves Fed ready to raise rates",
        "Trump meets Putin at G20 summit plans for comprehensive trade deal",
        "House passes 695 billion defense policy bill",
        "Feds Williams sees gradual rate hikes as key to further US growth",
        "US inflation unchanged in June retail sales fall",
        "Pentagon study declares American empire is collapsing",
        "Trumps lawyers looking to investigate Muellers team for conflicts of interest",
        "Scientists publish study on climate change impact on coastal regions",
        "New research shows benefits of Mediterranean diet for heart health",
        "Stock market closes higher after positive economic reports",
        "City council approves budget for infrastructure improvements",
        "International space station receives supply shipment from commercial rocket",
        "Researchers find link between exercise and improved cognitive function",
        "New species of frog discovered in Amazon rainforest",
        "Study finds correlation between sleep patterns and productivity",
        "Astronomers observe distant galaxy with unusual characteristics",
        "Local farmers collaborate to develop sustainable agricultural practices",
        "Recent archaeological dig uncovers ancient settlement artifacts",
        "Electric vehicle sales increase as battery technology improves",
        "European countries sign new trade agreement following negotiations",
        "Educators implement revised curriculum focusing on critical thinking",
        "Engineering team develops more efficient solar panel technology"
    ]
    
    # Create the data files using pandas to ensure proper CSV formatting
    fake_df = pd.DataFrame({
        'title': [f'Fake Title {i}' for i in range(len(fake_texts))],
        'text': fake_texts,
        'id': list(range(len(fake_texts)))
    })
    
    real_df = pd.DataFrame({
        'title': [f'Real Title {i}' for i in range(len(real_texts))],
        'text': real_texts,
        'id': list(range(100, 100 + len(real_texts)))
    })
    
    fake_df.to_csv(fake_path, index=False)
    real_df.to_csv(real_path, index=False)

# Load and preprocess the data
print("Loading and preprocessing data...")

try:
    # Load the datasets
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    
    # Check if datasets have expected columns
    if 'text' not in fake_df.columns or 'text' not in real_df.columns:
        # If columns are different, try to find text columns
        text_cols = [col for col in fake_df.columns if 'text' in col.lower() or 'content' in col.lower()]
        if text_cols:
            text_col = text_cols[0]
            fake_df = fake_df.rename(columns={text_col: 'text'})
            real_df = real_df.rename(columns={text_col: 'text'})
        else:
            # If no obvious text column, use the second column (often contains the text)
            fake_df = fake_df.rename(columns={fake_df.columns[1]: 'text'})
            real_df = real_df.rename(columns={real_df.columns[1]: 'text'})
    
    # Add labels: 'true' for fake news and 'false' for real news 
    # (this may seem counterintuitive but it's to match the existing app's classification)
    fake_df['Label'] = 'true'    # fake news
    real_df['Label'] = 'false'   # real news
    
    # Combine the datasets
    df = pd.concat([fake_df, real_df])
    
    # Drop NaN values
    df = df.dropna(subset=['text'])
    
    # Ensure we have enough data
    if len(df) < 20:
        raise ValueError("Not enough data after preprocessing")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")

    # Text preprocessing function
    def preprocess_text(text):
        try:
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
                
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and keep only alphabetic characters and spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize (simple split by whitespace)
            tokens = text.split()
            
            # Remove stopwords and short words
            tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
            
            # Join tokens back into a string
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return ""

    # Apply preprocessing to the text column
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['Label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a Naive Bayes classifier with adjusted class prior probabilities
    # This makes the model more balanced in its predictions
    print("Training model...")
    # Set class_prior to give slightly more weight to real news (makes the model less likely to flag everything as fake)
    model = MultinomialNB(class_prior=[0.45, 0.55])  # [fake, real] probability prior
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model and vectorizer saved successfully!")
    
    # Test with some example texts
    print("\nTesting model with example texts...")
    
    example_texts = [
        # Should be classified as fake news
        "BREAKING: Government officials announce unprecedented policy changes affecting millions",
        "Scientists discover miracle cure that big pharma doesn't want you to know about",
        "SHOCKING: Celebrity caught in scandalous affair with multiple partners",
        "URGENT: Secret government documents reveal conspiracy to control population",
        "BOMBSHELL: Famous politician hiding secret family in another country",
        
        # Should be classified as real news
        "New study shows correlation between diet and longevity in adults",
        "President meets with foreign leaders to discuss trade agreements",
        "Research indicates connection between sleep quality and mental health",
        "Local council approves new development project in city center",
        "Stock market shows modest gains following quarterly reports"
    ]
    
    example_processed = [preprocess_text(text) for text in example_texts]
    example_tfidf = vectorizer.transform(example_processed)
    example_predictions = model.predict(example_tfidf)
    example_proba = model.predict_proba(example_tfidf)
    
    print("\nExample Predictions:")
    for text, pred, proba in zip(example_texts, example_predictions, example_proba):
        confidence = max(proba) * 100
        result = "FAKE" if pred == "true" else "REAL"
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
        print("-" * 50)
    
except Exception as e:
    print(f"Error during model training: {e}")