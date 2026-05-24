

import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


data = {
    'email': [
        "Congratulations! You won $5000 click http://claim-money.com now",
        "Your bank account has been suspended verify immediately",
        "Meeting at 10 AM tomorrow in office",
        "Project submission deadline extended",
        "URGENT: Update your password here www.fakebank.com",
        "Hi friend, let's meet this weekend",
        "Free iPhone! Claim now at http://winiphone.com"
    ],

    'label': [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Safe",
        "Phishing"
    ]
}

df = pd.DataFrame(data)

# ==========================
# Custom Feature Extractor
# ==========================

class EmailFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, emails):

        features=[]

        for email in emails:

            url_count=len(
                re.findall(r'http[s]?://\S+|www\.\S+',email)
            )

            suspicious_words=[
                'urgent',
                'verify',
                'click',
                'free',
                'winner',
                'claim',
                'password'
            ]

            keyword_count=sum(
                word in email.lower()
                for word in suspicious_words
            )

            dollar_sign=email.count('$')

            features.append([
                url_count,
                keyword_count,
                dollar_sign
            ])

        return np.array(features)



pipeline=Pipeline([

('features',
 FeatureUnion([

('text',
 TfidfVectorizer()),

('custom',
 EmailFeatures())

 ])),

('classifier',
 RandomForestClassifier(
 n_estimators=100,
 random_state=42
 ))

])



X=df['email']
y=df['label']

X_train,X_test,y_train,y_test= train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



pipeline.fit(X_train,y_train)



predictions=pipeline.predict(X_test)



print("\nAccuracy:")
print(accuracy_score(y_test,predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))



new_email=[
"URGENT! Your account is blocked. Click http://secure-login.com"
]

result=pipeline.predict(new_email)

print("\nNew Email Prediction:")
print(result[0])
