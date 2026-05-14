import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

df = pd.read_csv('penguins_preprocessing/penguins_processed.csv')
X = df.drop('species', axis=1)
y = df['species']

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

mlflow.sklearn.save_model(model, "saved_model")
print("Model berhasil dilatih dan disimpan di direktori MLProject/saved_model/")
