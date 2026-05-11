import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

@st.cache_data
def load_and_build(path="movies.csv"):
    df = pd.read_csv(path)
    df["tags"] = df["genre"].str.replace(" ", "_") + " " + df["description"]
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["tags"])
    sim = cosine_similarity(matrix)
    return df, sim

def recommend(title, df, sim, n=5):
    idx = df[df["title"].str.lower() == title.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    results = []
    for i, score in scores:
        results.append({
            "title": df.iloc[i]["title"],
            "genre": df.iloc[i]["genre"],
            "match": f"{score:.0%}",
        })
    return results

df, sim = load_and_build()

st.title("Movie Recommender")
st.caption("Content-based filtering using TF-IDF + cosine similarity")

selected = st.selectbox("Pick a movie you like:", [""] + df["title"].tolist())

n = st.slider("Number of recommendations", 1, 10, 5)

if selected:
    results = recommend(selected, df, sim, n)
    if results:
        st.subheader(f"Because you liked **{selected}**:")
        for r in results:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{r['title']}**  \n`{r['genre']}`")
                col2.metric("Match", r["match"])
    else:
        st.error("Movie not found.")
else:
    st.info("Select a movie above to get recommendations.")

with st.expander("How it works"):
    st.markdown("""
1. Each movie's **genre + description** is combined into a text blob
2. **TF-IDF** converts that text into a numeric vector
3. **Cosine similarity** measures how close two movie vectors are
4. The top-N closest movies are returned as recommendations
    """)
