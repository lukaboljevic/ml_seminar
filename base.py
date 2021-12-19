import pandas as pd

from utils import COSINE, EUCLIDEAN, JACCARD
from utils import read_or_calc_similarity

def get_similar_games(df, title, mtx, metric):
    print(f"======= METRIC = {metric}, GAME = {title} =======\n")

    indices = pd.Series(df.index, index=df["name"]) # for each game, get it's index
    index = indices[title] # get the index for the title we're looking for
    similarity_scores = list(enumerate(mtx[index])) # get the similarity scores for this game with all others,
    # and enumerate them so we'll be able to detect which games it's closest to
    reverse = True
    if metric == EUCLIDEAN:
        reverse = False
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=reverse)
    similarity_scores = similarity_scores[1:21]

    game_indices = (i[0] for i in similarity_scores) # i = (index, score) so get just the indices
    print(df["name"].iloc[game_indices]) # retrieve those games
    print()
    print()

# print(early[early["name"].str.contains(title)]["name"])
# text = joined_df.loc[joined_df["appid"] == 6270]["about_the_game"].values[0]

df = pd.read_csv("sets/early_working_dataset.csv")
df["about_the_game"] = df["about_the_game"].fillna("")

metrics = [
    COSINE,
    EUCLIDEAN,
    # JACCARD,
]

description = True
games = [
    # "Team Fortress Classic",
    # "DOOM II",
    # "DmC: Devil May Cry",
    # "DiRT 4", 
    # "The Witcher 2: Assassins of Kings Enhanced Edition",
    # "Middle-earth™: Shadow of War™",
    # "Among Us",
    # "Trackmania United Forever Star Edition",
    "GRID 2",
]
for metric in metrics:
    if description:
        fname = f"intermediate_data/{metric}_sim_description.pickle"
    else:
        fname = f"intermediate_data/{metric}_sim.pickle"
    mtx = read_or_calc_similarity(fname, metric)
    for game in games:
        get_similar_games(df, game, mtx, metric)
    