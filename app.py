import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

KEYWORDS = {
    # MCU
    "Iron Man":                             "MCU Marvel superhero avengers",
    "The Avengers":                         "MCU Marvel superhero avengers loki",
    "Avengers Infinity War":                "MCU Marvel superhero avengers thanos",
    "Avengers Endgame":                     "MCU Marvel superhero avengers thanos",
    "Captain America Civil War":            "MCU Marvel superhero avengers",
    "Black Panther":                        "MCU Marvel superhero wakanda",
    "Spider-Man No Way Home":               "MCU Marvel superhero spiderman",
    "Doctor Strange":                       "MCU Marvel superhero magic sorcerer",
    "Thor Ragnarok":                        "MCU Marvel superhero thor asgard",
    "Guardians of the Galaxy":              "MCU Marvel superhero space",
    # DC
    "The Dark Knight":                      "DC batman gotham superhero",
    "Joker":                                "DC batman gotham villain",
    # Nolan
    "Inception":                            "nolan mindbending psychological",
    "Interstellar":                         "nolan space mindbending",
    "The Prestige":                         "nolan mindbending mystery magic",
    "Memento":                              "nolan mindbending psychological",
    "Dunkirk":                              "nolan war historical",
    # LOTR / fantasy epics
    "The Lord of the Rings Fellowship of the Ring": "LOTR tolkien hobbit fantasy epic",
    "The Lord of the Rings The Two Towers":         "LOTR tolkien hobbit fantasy epic",
    "The Lord of the Rings Return of the King":     "LOTR tolkien hobbit fantasy epic",
    "Harry Potter and the Sorcerers Stone":         "hogwarts magic wizard fantasy",
    # Star Wars
    "Star Wars A New Hope":                 "starwars jedi force galaxy space",
    "Star Wars The Empire Strikes Back":    "starwars jedi force galaxy space",
    "Rogue One":                            "starwars jedi force galaxy space",
    # Pixar
    "Toy Story":        "pixar animation family kids",
    "Finding Nemo":     "pixar animation family kids ocean",
    "Up":               "pixar animation family emotional",
    "WALL-E":           "pixar animation family robot",
    "Inside Out":       "pixar animation family emotional",
    "Coco":             "pixar animation family music",
    "The Incredibles":  "pixar animation family superhero",
    # Disney
    "The Lion King":    "disney animation family africa",
    "Frozen":           "disney animation family princess",
    "Moana":            "disney animation family ocean",
    # Bollywood / Hindi
    "Dilwale Dulhania Le Jayenge":  "bollywood hindi indian romance classic",
    "Kabhi Khushi Kabhie Gham":     "bollywood hindi indian family drama",
    "Kal Ho Naa Ho":                "bollywood hindi indian romance drama",
    "Jab We Met":                   "bollywood hindi indian romance comedy",
    "3 Idiots":                     "bollywood hindi indian comedy friendship college",
    "PK":                           "bollywood hindi indian comedy satire",
    "Dangal":                       "bollywood hindi indian sport wrestling",
    "Bajrangi Bhaijaan":            "bollywood hindi indian emotional drama",
    "Lagaan":                       "bollywood hindi indian cricket sport historical",
    "Sholay":                       "bollywood hindi indian classic action",
    "Gangs of Wasseypur":           "bollywood hindi indian crime gangster",
    "Gangs of Wasseypur 2":         "bollywood hindi indian crime gangster",
    "Andhadhun":                    "bollywood hindi indian thriller mystery",
    "Drishyam":                     "bollywood hindi indian thriller mystery crime",
    "Taare Zameen Par":             "bollywood hindi indian emotional family",
    "Queen":                        "bollywood hindi indian drama self-discovery",
    "Zindagi Na Milegi Dobara":     "bollywood hindi indian friendship travel",
    "Gully Boy":                    "bollywood hindi indian music rap",
    "Raazi":                        "bollywood hindi indian spy thriller",
    "URI The Surgical Strike":      "bollywood hindi indian war military",
    "Rockstar":                     "bollywood hindi indian music romance",
    "Tumbbad":                      "bollywood hindi indian horror fantasy",
    "Stree":                        "bollywood hindi indian horror comedy",
    "Bajirao Mastani":              "bollywood hindi indian historical romance war",
    "Padmaavat":                    "bollywood hindi indian historical romance war",
    "RRR":                          "indian telugu hindi action historical",
    "Baahubali The Beginning":      "indian telugu hindi action historical fantasy",
    "KGF Chapter 1":                "indian kannada hindi action crime",
    "Pathaan":                      "bollywood hindi indian action spy",
    "Mughal E Azam":                "bollywood hindi indian classic historical romance",
    "Devdas":                       "bollywood hindi indian classic romance tragedy",
    "Hum Dil De Chuke Sanam":       "bollywood hindi indian romance music",
    "Dil Chahta Hai":               "bollywood hindi indian friendship comedy",
    "Andaz Apna Apna":              "bollywood hindi indian comedy classic",
}

@st.cache_data
def load_and_build(path="movies.csv"):
    df = pd.read_csv(path)
    extra = df["title"].map(KEYWORDS).fillna("")
    # repeat genre 3x so franchise + genre both carry weight
    genre_tags = (df["genre"].str.replace(" ", "_") + " ") * 3
    df["tags"] = genre_tags + df["description"] + " " + extra
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
