import streamlit as st
import pandas as pd
import joblib
import time

@st.cache_data
def carregar_filmes():
    return pd.read_csv('data/movies.csv')

def carregar_avaliacoes():
    return pd.read_csv('data/ratings.csv')

@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_xgboost_filmes.pkl')

st.set_page_config(page_title="Recomendador Híbrido", layout="wide")

movies = carregar_filmes()
ratings = carregar_avaliacoes()
modelo_xgb = carregar_modelo()

st.sidebar.title("Área do Usuário")

aba_login, aba_cadastro = st.sidebar.tabs(["Login", "Novo Usuário"])

with aba_login:
    usuarios_disponiveis = sorted(ratings['userId'].unique().tolist())
    usuario_logado = st.selectbox("Selecione o seu ID:", usuarios_disponiveis)

with aba_cadastro:
    st.write("Crie um perfil para começar a avaliar.")
    novo_id = max(usuarios_disponiveis) + 1 if usuarios_disponiveis else 1
    if st.button(f"Criar Usuário ID: {novo_id}"):
        novo_registro = pd.DataFrame({'userId': [novo_id], 'movieId': [1], 'rating': [3.0], 'timestamp': [int(time.time())]})
        novo_registro.to_csv('data/ratings.csv', mode='a', header=False, index=False)
        st.success(f"Usuário {novo_id} criado! Atualize a página.")

st.title("Descubra Novos Filmes")

historico_usuario = ratings[ratings['userId'] == usuario_logado]
user_avg_rating = historico_usuario['rating'].mean()
qtd_avaliados = len(historico_usuario)

st.info(f"Logado como: Usuário {usuario_logado} | Filmes Avaliados: {qtd_avaliados} | Média de Rigor: {user_avg_rating:.2f} ⭐")

aba_avaliar, aba_recomendar, aba_historico = st.tabs(["⭐ Avaliar", "🎯 Recomendações", "📜 Meu Histórico"])

with aba_avaliar:
    st.markdown("### Avalie um filme que você assistiu")
    col_filme, col_nota, col_btn = st.columns([3, 1, 1])

    with col_filme:
        filme_selecionado_nome = st.selectbox("Busque no catálogo:", movies['title'].tolist())
    with col_nota:
        nota_dada = st.slider("Sua nota:", 0.5, 5.0, 4.0, 0.5)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Salvar Nota"):
            filme_alvo_id = movies[movies['title'] == filme_selecionado_nome]['movieId'].values[0]
            ja_avaliou = historico_usuario[historico_usuario['movieId'] == filme_alvo_id]
            
            if not ja_avaliou.empty:
                st.warning("Você já avaliou este filme!")
            else:
                nova_nota = pd.DataFrame({
                    'userId': [usuario_logado], 
                    'movieId': [filme_alvo_id], 
                    'rating': [nota_dada], 
                    'timestamp': [int(time.time())]
                })
                nova_nota.to_csv('data/ratings.csv', mode='a', header=False, index=False)
                st.success("Nota salva com sucesso!")
                time.sleep(1)
                st.rerun()
with aba_recomendar:
    st.markdown("### Receber Recomendações")
    if st.button("Gerar Recomendações Baseadas no meu Perfil"):
        if qtd_avaliados < 1:
            st.warning("Avalie pelo menos 1 filme para o sistema entender o seu perfil!")
        else:
            with st.spinner("Analisando padrões e cruzando com milhares de usuários..."):
                filme_alvo_id = historico_usuario.sort_values(by='rating', ascending=False).iloc[0]['movieId']
                generos_filme_alvo = movies[movies['movieId'] == filme_alvo_id]['genres'].values[0].split('|')
                
                matriz = ratings.pivot_table(index='userId', columns='movieId', values='rating')
                avaliacoes_alvo = matriz[filme_alvo_id]
                
                similaridade = matriz.corrwith(avaliacoes_alvo, method='pearson', min_periods=30)
                df_sim = pd.DataFrame(similaridade, columns=['Pearson']).dropna()
                candidatos = df_sim.join(movies.set_index('movieId'))
                
                candidatos = candidatos[candidatos['Pearson'] < 0.99]
                filmes_vistos = historico_usuario['movieId'].tolist()
                candidatos = candidatos[~candidatos.index.isin(filmes_vistos)]
                
                resultados = []
                
                for idx, row in candidatos.iterrows():
                    generos_cand = row['genres'].split('|')
                    intersecao = set(generos_filme_alvo).intersection(set(generos_cand))
                    
                    req = 1 if len(generos_filme_alvo) == 1 else 2
                    if len(intersecao) >= req:
                        
                        movie_avg = ratings[ratings['movieId'] == idx]['rating'].mean()
                        movie_pop = ratings[ratings['movieId'] == idx]['rating'].count()
                        
                        features = pd.DataFrame([{
                            'user_avg_rating': user_avg_rating,
                            'movie_avg_rating': movie_avg,
                            'movie_popularity': movie_pop,
                            'Sci-Fi': 1 if 'Sci-Fi' in generos_cand else 0,
                            'Animation': 1 if 'Animation' in generos_cand else 0,
                            'Action': 1 if 'Action' in generos_cand else 0,
                            'Musical': 1 if 'Musical' in generos_cand else 0,
                            'Horror': 1 if 'Horror' in generos_cand else 0,
                            'IMAX': 1 if 'IMAX' in generos_cand else 0,
                            'Children': 1 if 'Children' in generos_cand else 0
                        }])
                        
                        prob_match = modelo_xgb.predict_proba(features)[0][1] * 100
                        resultados.append({
                            'title': row['title'],
                            'genres': row['genres'],
                            'pearson': row['Pearson'],
                            'nota_geral': movie_avg,
                            'match': prob_match
                        })
                
                df_resultados = pd.DataFrame(resultados)
                if not df_resultados.empty:
                    df_resultados = df_resultados.sort_values(by=['pearson', 'nota_geral'], ascending=[False, False]).head(10)
                    
                    st.success("Recomendações geradas com sucesso!")
                    for i, row in df_resultados.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(row['title'])
                                st.write(f"🎭 Gêneros: {row['genres']}")
                                st.write(f"📊 Similaridade Colaborativa: {row['pearson']*100:.1f}% | Nota Geral: {row['nota_geral']:.1f}")
                            with col2:
                                cor = "🟢" if row['match'] >= 50 else "🟡"
                                st.metric(label=f"{cor} XGBoost Match", value=f"{row['match']:.1f}%")
                            st.divider()
                else:
                    st.warning("Não encontramos correspondências exatas com base nesse filme e nos filtros rigorosos aplicados.")
with aba_historico:
    if historico_usuario.empty:
        st.info("Você ainda não tem um histórico de filmes avaliados.")
    else:
        historico_completo = historico_usuario.merge(movies, on='movieId')
        
        st.markdown("#### 🏆 Seus Gêneros Favoritos")
        historico_explodido = historico_completo.copy()
        historico_explodido['genres'] = historico_explodido['genres'].str.split('|')
        historico_explodido = historico_explodido.explode('genres')
        
        stats_genero = historico_explodido.groupby('genres').agg(
            Nota_Media=('rating', 'mean'),
            Qtd_Filmes=('rating', 'count')
        ).reset_index()
        
        stats_genero = stats_genero.sort_values(by=['Nota_Media', 'Qtd_Filmes'], ascending=[False, False])
        stats_genero.columns = ['Gênero', 'Nota Média', 'Quantidade Assistida']
        st.dataframe(stats_genero.head(5), use_container_width=True, hide_index=True)
        
        st.markdown("#### 🎬 Todos os Filmes Avaliados")
        historico_exibicao = historico_completo[['title', 'genres', 'rating']].sort_values(by='rating', ascending=False)
        historico_exibicao.columns = ['Título do Filme', 'Gêneros', 'Sua Nota']
        st.dataframe(historico_exibicao, use_container_width=True, hide_index=True)