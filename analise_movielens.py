import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def executar_analise():
    print("Iniciando a leitura dos CSVs...\n")
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        print("Erro: Certifique-se de que 'movies.csv' e 'ratings.csv' estão na mesma pasta que este script.")
        return
    print("=== 1. VISÃO GERAL DOS DADOS ===")
    print(f"Total de Filmes no catálogo: {movies.shape[0]}")
    print(f"Total de Avaliações (Notas) registradas: {ratings.shape[0]}")
    print(f"Total de Usuários únicos: {ratings['userId'].nunique()}\n")
    print("=== 2. ANÁLISE DE TODAS AS NOTAS ===")
    media_global = ratings['rating'].mean()
    mediana_global = ratings['rating'].median()
    print(f"Nota Média Global: {media_global:.2f} estrelas")
    print(f"Nota Mediana Global: {mediana_global:.2f} estrelas")
    contagem_notas = ratings['rating'].value_counts().sort_index(ascending=False)
    print("\nDistribuição exata das notas (da maior para a menor):")
    for nota, quantidade in contagem_notas.items():
        porcentagem = (quantidade / len(ratings)) * 100
        print(f"Nota {nota}: {quantidade} avaliações ({porcentagem:.1f}%)")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=ratings, x='rating', palette='viridis')
    plt.title('Distribuição de Todas as Notas (MovieLens)')
    plt.xlabel('Nota (Estrelas)')
    plt.ylabel('Quantidade de Avaliações')
    plt.savefig('grafico_notas.png')
    print("\n-> Gráfico salvo com sucesso: 'grafico_notas.png'\n")

    print("=== 3. ANÁLISE DE TODOS OS GÊNEROS ===")
    
    todos_generos_listas = movies['genres'].str.split('|')
    
    lista_plana_generos = [genero for sublista in todos_generos_listas for genero in sublista]
    contagem_generos = Counter(lista_plana_generos)
    
    df_generos = pd.DataFrame.from_dict(contagem_generos, orient='index', columns=['Quantidade'])
    df_generos = df_generos.sort_values(by='Quantidade', ascending=False)
    
    print("Quantidade de filmes por gênero:")
    print(df_generos.to_string())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=df_generos['Quantidade'], y=df_generos.index, palette='mako')
    plt.title('Volume de Filmes por Gênero')
    plt.xlabel('Número de Filmes')
    plt.ylabel('Gênero')
    plt.tight_layout()
    plt.savefig('grafico_generos.png')
    print("\n-> Gráfico salvo com sucesso: 'grafico_generos.png'\n")

    print("=== 4. OS 10 FILMES COM MAIS AVALIAÇÕES ===")
    avaliacoes_por_filme = ratings.groupby('movieId').size().reset_index(name='Total_Avaliacoes')
    
    filmes_populares = pd.merge(avaliacoes_por_filme, movies[['movieId', 'title']], on='movieId')
    filmes_populares = filmes_populares.sort_values(by='Total_Avaliacoes', ascending=False).head(10)
    
    print(filmes_populares[['title', 'Total_Avaliacoes']].to_string(index=False))

if __name__ == "__main__":
    executar_analise()