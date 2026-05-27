import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

print("Carregando dados da pasta 'data'...")
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

print("Criando variáveis (Feature Engineering)...")
ratings['gostou'] = (ratings['rating'] >= 3.5).astype(int)

user_avg = ratings.groupby('userId')['rating'].mean().reset_index(name='user_avg_rating')
movie_avg = ratings.groupby('movieId')['rating'].mean().reset_index(name='movie_avg_rating')
movie_pop = ratings.groupby('movieId').size().reset_index(name='movie_popularity')

df = ratings.merge(user_avg, on='userId').merge(movie_avg, on='movieId').merge(movie_pop, on='movieId')
df = df.merge(movies[['movieId', 'genres']], on='movieId')

generos_alvo = ['Sci-Fi', 'Animation', 'Action', 'Musical', 'Horror', 'IMAX', 'Children']
for genero in generos_alvo:
    df[genero] = df['genres'].apply(lambda x: 1 if genero in x else 0)

features = ['user_avg_rating', 'movie_avg_rating', 'movie_popularity'] + generos_alvo
X = df[features]
y = df['gostou']

# 6. Separação Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treinando o modelo XGBoost...")
modelo = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1] 

print("\n=== MÉTRICAS PARA O RESUMO EXPANDIDO (ATIVIDADE 1) ===")
print(f"Acurácia:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print("======================================================\n")

caminho_modelo = 'modelo_xgboost_filmes.pkl'
joblib.dump(modelo, caminho_modelo)
print(f"✅ Modelo salvo com sucesso: '{caminho_modelo}'")