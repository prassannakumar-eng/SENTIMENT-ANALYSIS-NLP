import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    "review": [
        "I love this product",
        "This is terrible",
        "Amazing experience",
        "Worst service ever",
        "Very happy with the purchase",
        "Not good at all"
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df['review'] = df['review'].str.lower()
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
text = ["This product is amazing"]
text_tfidf = vectorizer.transform(text)
prediction = model.predict(text_tfidf)
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
