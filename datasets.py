import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import os
import pickle
from scipy.sparse import hstack

CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
english_stopwords = stopwords.words('english')

def preprocess_text_tfidf(text):
    # can probably change the order of the operations but it's fine atm
    text = re.sub(CLEANR, " ", text)
    text = re.sub("\s\s+", " ", text)
    text = text.lower()
    text = re.sub("\W", " ", text)
    text = " ".join([word for word in text.split() if word not in english_stopwords])

    non_ascii = 0
    for word in text.split():
        if not word.isascii(): non_ascii += 1
    if non_ascii >= 40:
        return None
    else:
        return text

def preprocess_text_countvec(text):
    # to lower, remove stopwords, remove non word chars
    text = re.sub("\W", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in english_stopwords])
    return text

def remove_semicolons(text):
    return text.replace(";", " ")

# def remove_edition_from_game_name(text):
#     # given a game name, remove X Edition from it's name cause
#     # that's just useless info really
#     return text.split()

def make_usable_dataset():
    main_columns = [
        "appid",
        "english",
        "name",
        "developer",
        "steamspy_tags",
        "positive_ratings",
        "negative_ratings"
    ]
    desc_columns = [
        "steam_appid",
        "about_the_game"
    ]
    main_df = pd.read_csv("sets/steam.csv", usecols=main_columns)
    desc_df = pd.read_csv("sets/steam_description_data.csv", usecols=desc_columns)
    desc_df.rename(columns={"steam_appid": "appid",}, inplace=True)
    joined_df = main_df.merge(desc_df, on="appid")

    joined_df = joined_df[joined_df["english"] == 1].reset_index(drop=True) # remove games which don't have english translation
    joined_df = joined_df.drop("english", axis=1) # we don't need it anymore

    joined_df["about_the_game"] = joined_df["about_the_game"].apply(preprocess_text_tfidf)
    joined_df = joined_df[pd.notnull(joined_df["about_the_game"])]
    joined_df["developer"] = joined_df["developer"].apply(remove_semicolons)
    joined_df["steamspy_tags"] = joined_df["steamspy_tags"].apply(remove_semicolons)
    # Unite tags, name, developer
    joined_df["recommendation_info"] = joined_df["name"] + " " + joined_df["developer"] + " " + joined_df["steamspy_tags"]
    joined_df["recommendation_info"] = joined_df["recommendation_info"].apply(preprocess_text_countvec)

    # Save
    joined_df.to_csv("sets/early_working_dataset.csv", index=False)

def vectorize(df): 
    # Perform count vectorization on name, tags, developer
    fname = "intermediate_data/countvec_matrix.pickle"
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            count_matrix = pickle.load(f)
            print(count_matrix.shape)
    else:
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform(df["recommendation_info"])
        with open(fname, "wb") as f:
            pickle.dump(count_matrix, f)
    print("Count vectorization done!")

    # Perform TFIDF vectorization on the game description
    fname = "intermediate_data/tfidf_matrix.pickle"
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            tfidf_matrix = pickle.load(f)
            print(tfidf_matrix.shape)
    else:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["about_the_game"])
        with open(fname, "wb") as f:
            pickle.dump(tfidf_matrix, f)
    print("TFIDF vectorization done!")

    # First compute the cosine similarity matrix for name, tags and developer
    fname = "intermediate_data/cosine_sim.pickle"
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            cosine_sim = pickle.load(f)
            print(cosine_sim.shape)
    else:
        cosine_sim = linear_kernel(count_matrix, count_matrix)
        with open(fname, "wb") as f:
            pickle.dump(cosine_sim, f)
    print("Cosine similarity matrix for name, tags and the developer is done!")

    # Then, join the count matrix with the tfidf matrix so that we can take
    # descriptions into account too!
    joined_matrices = hstack([count_matrix, tfidf_matrix])
    fname = "intermediate_data/cosine_sim_description.pickle"
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            cosine_sim_description = pickle.load(f)
            print(cosine_sim_description.shape)
    else:
        cosine_sim_description = linear_kernel(joined_matrices, joined_matrices)
        with open(fname, "wb") as f:
            pickle.dump(cosine_sim_description, f)
    print("Cosine similarity matrix for game description is done!")

def get_similar_games(df, title, cosine_mtx):
    print(f"======= GAME = {title} =======\n")

    indices = pd.Series(df.index, index=df["name"])
    index = indices[title]
    similarity_scores = list(enumerate(cosine_mtx[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:16]

    game_indices = (i[0] for i in similarity_scores)
    print(df["name"].iloc[game_indices])
    print()
    print()

# print(early[early["name"].str.contains(title)]["name"])
# text = joined_df.loc[joined_df["appid"] == 6270]["about_the_game"].values[0]

# make_usable_dataset()
# print("Done with dataset")

df = pd.read_csv("sets/early_working_dataset.csv")
df["about_the_game"] = df["about_the_game"].fillna("")

# # vectorize(df)
# # print("Done with vectorization")

fname = "intermediate_data/cosine_sim.pickle"
with open(fname, 'rb') as f:
    cosine_sim = pickle.load(f)

fname = "intermediate_data/cosine_sim_description.pickle"
with open(fname, 'rb') as f:
    cosine_sim_desc = pickle.load(f)

games = [
    "Team Fortress Classic",
    "DOOM II",
    "DmC: Devil May Cry",
    "DiRT 4", 
    "The Witcher 2: Assassins of Kings Enhanced Edition",
    "Middle-earth™: Shadow of War™",
    "Among Us"
]
for game in games:
    # get_similar_games(df, game, cosine_sim)
    get_similar_games(df, game, cosine_sim_desc)
    