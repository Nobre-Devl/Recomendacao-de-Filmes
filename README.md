# 🎬 Sistema Híbrido de Recomendação de Filmes

Projeto desenvolvido como parte dos estudos de **Mineração de Dados**, focado na construção de um motor de recomendação inteligente utilizando a base de dados do MovieLens.

O sistema não entrega apenas os filmes mais populares, mas busca a **serendipidade** (descobertas inesperadas e altamente relevantes) cruzando matemática estatística com aprendizado de máquina.

## 🧠 Como o Sistema Funciona?

A arquitetura do motor de recomendação baseia-se em um **Pipeline Híbrido de Duas Etapas**:

1. **Etapa A: Filtragem Colaborativa (Item-to-Item)**
   * Utiliza a **Correlação de Pearson** para mapear a distância matemática entre os comportamentos de milhares de usuários. 
   * Quando o usuário escolhe um filme base, o sistema procura quais outros filmes receberam notas em padrões estatisticamente idênticos.
   * *Filtro de Ruído:* O sistema exige uma interseção mínima de 30 usuários diferentes (`min_periods=30`) para validar a similaridade, eliminando falsos positivos e o viés de pequena amostragem.

2. **Etapa B: Validação Preditiva (Machine Learning)**
   * Os candidatos que sobrevivem à matemática pura passam por um modelo **XGBoost** pré-treinado.
   * O modelo cruza o histórico do usuário (rigor da nota média), a recepção global do filme e características categóricas (Gêneros) para calcular o **Match (%)** — a probabilidade real do usuário logado gostar da indicação.

3. **Regras de Negócio Rígidas**
   * **Guardião de Gênero:** O filme recomendado deve compartilhar pelo menos 1 a 2 gêneros com o filme base.
   * **Filtro de Ineditismo:** Filmes que o usuário já assistiu e avaliou são ocultados automaticamente da lista final.

---

## 📂 Estrutura de Pastas Exigida

Para que o projeto funcione corretamente, a sua pasta deve estar organizada exatamente desta forma:

```text
meu_projeto/
│
├── data/
│   ├── movies.csv        # Tabela de filmes (MovieLens)
│   └── ratings.csv       # Tabela de avaliações (MovieLens)
│
├── analise_movielens.py  # Script de Análise Exploratória (EDA)
├── treinar_modelo.py     # Script de treinamento do XGBoost
├── app.py                # Interface visual em Streamlit
└── README.md             # Este arquivo
```

---

## 🚀 Requisitos e Passo a Passo para Execução

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.8+** instalado em sua máquina. 
No seu terminal, instale todas as bibliotecas necessárias executando o comando abaixo:
```bash
pip install pandas matplotlib seaborn xgboost scikit-learn joblib streamlit
```

### 2. Análise Exploratória (Opcional, mas recomendado)
Para gerar métricas globais e visualizar gráficos sobre a distribuição de notas e o peso dos gêneros na base de dados, execute:
```bash
python analise_movielens.py
```
*Isso gerará os gráficos `grafico_notas.png` e `grafico_generos.png` na raiz do projeto.*

### 3. Treinar e Exportar o Modelo (Obrigatório)
Antes de rodar a interface web, o sistema precisa criar o seu "cérebro preditivo" exportando o modelo treinado. Execute o script de treinamento:
```bash
python treinar_modelo.py
```
*O script fará o Feature Engineering, avaliará as métricas de qualidade (Acurácia, Precisão, F1-Score, ROC AUC) e exportará o arquivo `modelo_xgboost_filmes.pkl` na raiz do projeto.*

### 4. Iniciar a Interface Web
Com o modelo `.pkl` treinado e salvo na pasta, suba o aplicativo visual rodando o seguinte comando no terminal:
```bash
python -m streamlit run app.py
```
*Uma aba abrirá automaticamente no seu navegador padrão (geralmente no endereço `http://localhost:8501`).*

---

## 💻 Funcionalidades da Interface

O painel visual (*Front-End*) foi construído de forma limpa usando Streamlit e divide-se em abas principais:

* **Menu Lateral (Login/Cadastro):** Permite que o usuário cadastre um novo ID dinamicamente e faça login para acessar seu perfil.
* **⭐ Avaliar:** Permite que o usuário logado dê notas para os filmes do catálogo, alimentando o banco de dados `ratings.csv` em tempo real e reajustando o modelo ao seu perfil.
* **🎯 Recomendações:** A engrenagem principal. O usuário clica no botão e o sistema cruza o seu histórico com a base do MovieLens, exibindo um Top 10 de recomendações com ordenação matemática decrescente e indicativos visuais de Match (Verde/Amarelo).
* **📜 Meu Histórico:** Um painel analítico individual que exibe os gêneros favoritos do usuário (ordenados por nota média) e o registro histórico completo dos filmes avaliados por ele.
