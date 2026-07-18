import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json

df = pd.read_csv('ml_model/Training.csv')
if 'Unnamed: 133' in df.columns:
    df = df.drop('Unnamed: 133', axis=1)

df['prognosis'] = df['prognosis'].str.strip()
X = df.drop('prognosis', axis=1)
y = df['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)

all_symptoms = list(X.columns)

def test_model(model, symps):
    vec = np.zeros(len(all_symptoms))
    for s in symps:
        if s in all_symptoms:
            vec[all_symptoms.index(s)] = 1
    probs = model.predict_proba([vec])[0]
    top_indices = np.argsort(probs)[::-1][:3]
    for idx in top_indices:
        print(f"  {model.classes_[idx]}: {probs[idx]*100:.1f}%")

print("Test 1: itching, sweating, back_pain, constipation, coma")
symps1 = ['itching', 'sweating', 'back_pain', 'constipation', 'coma']
print("RF:")
test_model(rf, symps1)
print("NB:")
test_model(nb, symps1)

print("\nTest 2: skin_rash, stomach_pain, headache")
symps2 = ['skin_rash', 'stomach_pain', 'headache']
print("RF:")
test_model(rf, symps2)
print("NB:")
test_model(nb, symps2)
