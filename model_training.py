import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('news.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_title'] = df['title'].apply(clean_text)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    df[['cleaned_text', 'cleaned_title']], df['label'], test_size=0.2, random_state=42
)
