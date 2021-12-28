import os
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from requests import get

COSINE = 'cosine'
EUCLIDEAN = 'euclidean'
JACCARD = 'jaccard'
CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});") # for removing tags https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
english_stopwords = stopwords.words('english')

################################ GENERAL FUNCTIONS ################################

def find_closest_title(game_names, title):
    """
    Returns the most likely game name for this particular title

    Parameters
    ----------
    game_names: pd.DataFrame
        A dataframe that has at least a column with all the names

    title: str

    """
    all_games = game_names["name"].values
    matches = []
    for game in all_games:
        ratio = fuzz.ratio(title.lower(), game.lower())
        if ratio >= 60:
            matches.append((game, ratio))
    if not matches:
        return None
    matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return matches[0][0]

def percentage(f):
    """
    Return the percentage version of a float f
    """
    return round(f * 100, 3)

################################ DATASET RELATED FUNCTIONS ################################

def preprocess_text_tfidf(text):
    """
    Preprocess text for TfIdf vectorization
    """
    text = re.sub(CLEANR, " ", text) # remove tags and stuff
    text = re.sub("\s\s+", " ", text) # remove blancos
    text = text.lower() # to lowercase
    text = re.sub("\W", " ", text) # remove non word characters
    text = " ".join([word for word in text.split() if word not in english_stopwords]) # remove stop words

    # If it has a lot on non ascii characters like ü or whatever, it's probably not good to train on this
    non_ascii = 0
    for word in text.split():
        if not word.isascii(): non_ascii += 1
    if non_ascii >= 40:
        return None
    else:
        return text

def preprocess_text_countvec(text):
    """
    Preprocess text for count vectorization
    """
    # to lower, remove stopwords, remove non word chars
    text = re.sub("\W", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in english_stopwords])
    return text

def remove_semicolons(text):
    return text.replace(";", " ")

def make_content_based_dataset(main_set):
    """
    Make a dataset to work on for the content based filtering approach
    """
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
    main_df = pd.read_csv(f"sets/{main_set}.csv", usecols=main_columns)
    desc_df = pd.read_csv("sets/steam_description_data.csv", usecols=desc_columns)
    desc_df.rename(columns={"steam_appid": "appid",}, inplace=True) # make it the same name
    joined_df = main_df.merge(desc_df, on="appid")

    joined_df = joined_df[joined_df["english"] == 1].reset_index(drop=True) # remove games which don't have english translation
    joined_df = joined_df.drop("english", axis=1) # we don't need this column anymore

    joined_df["about_the_game"] = joined_df["about_the_game"].apply(preprocess_text_tfidf) # get the game description column ready for training
    joined_df = joined_df[pd.notnull(joined_df["about_the_game"])] # there may be nulls so remove them
    # Apply simple processing for developer and steam tags columns - this is done because that's how the dataset is made
    joined_df["developer"] = joined_df["developer"].apply(remove_semicolons) 
    joined_df["steamspy_tags"] = joined_df["steamspy_tags"].apply(remove_semicolons)
    # Unite tags, name, developer and use it later for recommending
    joined_df["recommendation_info"] = joined_df["name"] + " " + joined_df["developer"] + " " + joined_df["steamspy_tags"]
    joined_df["recommendation_info"] = joined_df["recommendation_info"].apply(preprocess_text_countvec)

    # Save
    joined_df.to_csv("sets/content_based_dataset.csv", index=False)


"""
I will also convert the positive and negative ratings into a percentage:
positive / (positive + negative)

Maybe a good idea would be to first remove games with less than like 5000 or so ratings
just because if they have like 4000 positive and 1000 negative ratings
"""

def make_collaborative_based_dataset(main_set): 
    """
    Make a dataset to work on for the collaborative based filtering approach
    """
    main_columns = [
        "appid",
        "name",
        "english",
        "positive_ratings",
        "negative_ratings"
    ]
    main_df = pd.read_csv(f"sets/{main_set}.csv", usecols=main_columns)
    main_df = main_df[main_df["english"] == 1].reset_index(drop=True) # remove games which don't have english translation
    main_df = main_df.drop("english", axis=1) # we don't need this column anymore

    main_df["total_ratings"] = main_df["positive_ratings"] + main_df["negative_ratings"]
    main_df = main_df[main_df["total_ratings"] >= 1000] # remove little rated games
    main_df["rating_percentage"] = main_df["positive_ratings"] / main_df["total_ratings"]

    names = main_df[["appid", "name"]].copy()
    names.to_csv("sets/collaborative_based_dataset_names.csv", index=False)

    main_df = main_df.drop("name", axis=1) # we don't need this column anymore
    main_df.to_csv("sets/collaborative_based_dataset.csv", index=False)

################################ CONTENT-BASED FILTERING RELATED FUNCTIONS ################################

def jaccard(vec1, vec2):
    """
    Jaccard similarity between two vectors
    """
    intersection = np.logical_and(vec1, vec2)
    union = np.logical_or(vec1, vec2)
    return intersection.sum() / float(union.sum())

def read_or_calc_vectorization_matrix(df, filename, vectorization, column, overwrite=False):
    """
    Given a data frame, perform the necessary vectorization (Count or TfIdf)
    on the given column (either read or do the calculation)

    Parameters
    ----------
    df: pd.DataFrame
        A data frame containing the data related to games
    filename: str
        The file to save or read the vectorization matrix from
    vectorization: str
        Which vectorization method to use - TfIdf or Count
    column: str
        Which column to use from the data frame
    overwrite: boolean
        If True, and the file exists, will overwrite the contents. Default is False

    Returns
    -------
    A vectorization matrix for the given vectorization method on the given column (either calculated and saved, or just read from a file)
    """
    if os.path.exists(filename):
        if overwrite:
            # If it exists, but we wanna overwrite, continue
            pass
        else: 
            # If we don't want to overwrite, load and return the matrix
            with open(filename, 'rb') as f:
                mtx = pickle.load(f)
                return mtx
    
    # Else fit and transform using one of the vectorization methods
    vec = TfidfVectorizer() if vectorization == "tfidf" else CountVectorizer()
    mtx = vec.fit_transform(df[column])
    with open(filename, "wb") as f:
        pickle.dump(mtx, f)
    return mtx

def read_or_calc_similarity(filename, metric, mtx=None, overwrite=False):
    """
    Read a similarity matrix for a given metric, or calculate it and save it.

    Parameters
    ----------
    filename: str
        The file to save or read the similarity matrix from
    metric: str
        What metric (cosine, euclidean)
    mtx:
        A matrix fitted and transformed by CountVectorizer or TfIdfVectorizer
    overwrite: boolean
        If True, and the file exists, will overwrite the contents. Default is False

    Returns
    -------
    A similarity matrix for the given metric (either calculated and saved, or just read from a file)
    """
    if os.path.exists(filename):
        if overwrite:
            # If it exists, but we wanna overwrite, continue
            pass
        else:
            with open(filename, 'rb') as f:
                mtx = pickle.load(f)
                return mtx
    
    # Actually calculate the similarity matrix
    if mtx is None:
        raise ValueError("The matrix has not yet been computed and saved, but it is not given as a param")
    sim = None
    if metric == COSINE:
        sim = linear_kernel(mtx)
    elif metric == EUCLIDEAN:
        sim = euclidean_distances(mtx)
    elif metric == JACCARD:
        # This takes so much time like jesus
        n = mtx.shape[0]
        sim = np.ndarray(shape=(n, n), dtype=mtx.dtype)
        for i in range(n):
            for j in range(n):
                if i <= j:
                    jac = jaccard(mtx[i], mtx[j])
                    sim[i][j] = jac
                    sim[j][i] = jac
    with open(filename, "wb") as f:
        pickle.dump(sim, f)
    return sim

def vectorize_and_similaritize(df, metrics):
    """
    Calculate the vectorization matrices using TfIdfVectorizer or CountVectorizer, 
    or read them, and calculate/read the similarity matrices for the given distance
    metrics.
    """

    # Perform count vectorization on name, tags, developer (ie recommendation info)
    # as these are not "long text" based attributes.
    filename = "intermediate_data/countvec_matrix.pickle"
    count_matrix = read_or_calc_vectorization_matrix(df, filename, "count", "recommendation_info")#, overwrite=True)
    print("Count vectorization done!")

    # Perform TFIDF vectorization on the game description as this is a "long text" attribute.
    filename = "intermediate_data/tfidf_matrix.pickle"
    tfidf_matrix = read_or_calc_vectorization_matrix(df, filename, "tfidf", "about_the_game")#, overwrite=True)
    print("TFIDF vectorization done!")

    # Join the count matrix with the tfidf matrix so that we can take
    # descriptions into account too when recommending!
    joined_matrices = hstack([count_matrix, tfidf_matrix])

    # Compute all the similarity matrices for the count matrix, and joined matrices
    for metric in metrics:
        filename = f"intermediate_data/{metric}_sim.pickle"
        if metric == JACCARD:
            mtx = read_or_calc_similarity(filename, metric, count_matrix.toarray(), overwrite=True)
        else:
            mtx = read_or_calc_similarity(filename, metric, count_matrix, overwrite=True)
        print(f"CountVectorizer matrix for {metric} metric done!")

        filename = f"intermediate_data/{metric}_sim_description.pickle"
        mtx = read_or_calc_similarity(filename, metric, joined_matrices, overwrite=True)
        print(f"Joined matrices for {metric} metric done!")

################################ ITEM-BASED COLLABORATIVE FILTERING RELATED FUNCTIONS ################################

def train_model():
    # Read the set
    all_ratings = pd.read_csv("sets/collaborative_based_dataset.csv")

    # Extract raw rating percentages, form into an N x 1 matrix
    # so we can train it using KNN
    raw_rating_perc = all_ratings["rating_percentage"].values
    n = len(raw_rating_perc)
    perc_matrix = np.array(raw_rating_perc).reshape(n, 1)

    # In the matrix made above, the entry at index i corresponds to the
    # rating percentage of a game that was in the i-th row in the dataset
    # (and that game has its appid i.e. game id)
    # So create this mapping and the reverse mapping for later
    ids = all_ratings["appid"]
    index_to_appid = {
        index: appid for index, appid in enumerate(ids)
    }
    appid_to_index = {
        appid: index for index, appid in index_to_appid.items()
    }

    # Make the model
    knn = NearestNeighbors(metric="euclidean", algorithm="brute").fit(perc_matrix)

    return knn, perc_matrix, index_to_appid, appid_to_index


# from time import perf_counter
# start = perf_counter()
# make_content_based_dataset("steam_updated")
# df = pd.read_csv("sets/content_based_dataset.csv")
# df["about_the_game"] = df["about_the_game"].fillna("")
# vectorize_and_similaritize(df, [EUCLIDEAN])
# print(perf_counter() - start)

# make_collaborative_based_dataset("steam_updated")