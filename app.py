#!/usr/bin/env python3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

def get_dataset():
    data = {
        'text': [
            # Positive samples 1
            'I love this movie, it is amazing!', 
            'The food was delicious and great.', 
            'Fantastic service, highly recommend.',
            'This is the best day ever.', 
            'Absolutely beautiful and wonderful experience.',
            'Truly impressive results and super friendly team.',
            'An astounding achievement, well done!',
            'Exceptional quality and very fast delivery.',
            'Highly satisfied with the overall experience.',
            'Brilliant performance and lovely atmosphere.',
            'Perfect solution, works like a charm!',
            'I am so happy with this purchase.',
            'Delightful treat, worth every penny.',
            'Great value for money, excellent design.',
            'Super helpful staff and quick resolution.',

            # Negative samples 0
            'I hate this product, it is terrible.', 
            'The worst service I have ever seen.', 
            'Horrible experience and very bad.',
            'This is completely broken and useless.', 
            'Very disappointed with the quality.',
            'Waste of money, do not buy this.',
            'Extremely poor customer support and rude staff.',
            'Totally defective and arrived damaged.',
            'Unacceptable behavior and frustrating delay.',
            'Awful product, completely stopped working.',
            'Regret buying this, zero value provided.',
            'Very bad experience, completely dissatisfied.',
            'Painful to use and very confusing interface.',
            'Slow performance, crashes all the time.',
            'Low quality materials, broke on the first day.'
        ],
        'sentiment': [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  
        ]
    }
    return pd.DataFrame(data)

def train_model(df):
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    
    combined_features = FeatureUnion([
        ('word_features', word_vectorizer),
        ('char_features', char_vectorizer)
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        ('classifier', LogisticRegression())
    ])
    
    pipeline.fit(df['text'], df['sentiment'])
    return pipeline

def main():
    df = get_dataset()
    model_pipeline = train_model(df)
    
    print("Model trained !\n")
    print("--- Hybrid Sentiment Analyzer  ---")
    
    while True:
        user_sentence = input("Enter a sentence to test (or enter '0' to exit): ").strip()
        
        if user_sentence == '0':
            print("\nExiting analyzer. Goodbye!")
            break
            
        if not user_sentence:
            continue
            
        prediction = model_pipeline.predict([user_sentence])
        
        if prediction[0] == 1:
            print("Result: Positive Sentiment\n")
        else:
            print("Result: Negative Sentiment\n")

if __name__ == '__main__':
    main()