import pandas as pd

from utils import COSINE, EUCLIDEAN, JACCARD
from utils import read_or_calc_similarity, train_model, find_closest_title, percentage

def content_based_similar_games(title, mtx, metric, num_recommendations):
    
    # Read necessary set
    df = pd.read_csv("sets/content_based_dataset.csv")
    df["about_the_game"] = df["about_the_game"].fillna("") # for some reason I need to do this

    # Get the most likely title for the input
    temp = pd.DataFrame(df["name"], columns=["name"])
    closest_title = find_closest_title(temp, title)
    if not closest_title:
        print(f"No title found for \"{title}\" !!!!")
        return

    print(f"======= GAME = {closest_title} =======\n")
    print(f"======= METRIC = {metric} =======\n")

    indices = pd.Series(df.index, index=df["name"]) # for each game, get it's index
    index = indices[closest_title] # get the index for the title we're looking for
    similarity_scores = list(enumerate(mtx[index])) # get the similarity scores for this game with all others,
    # and enumerate them so we'll be able to detect which games it's closest to
    reverse = True
    if metric == EUCLIDEAN:
        reverse = False
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=reverse)[1:(num_recommendations+1)]

    game_indices = (i[0] for i in similarity_scores) # i = (index, score) so get just the indices
    # Time to retrieve and print out the recommendations
    for recommended_game in df["name"].iloc[game_indices].values:
        print(f"\t{recommended_game}")

def collaborative_based_similar_games(title, num_recommendations):
    
    # Obtain the necessary data
    all_ratings = pd.read_csv("sets/collaborative_based_dataset.csv")
    game_names = pd.read_csv("sets/collaborative_based_dataset_names.csv")
    knn, perc_matrix, index_to_appid, appid_to_index = train_model()

    # Find the most likely title for this input title
    closest_title = find_closest_title(game_names, title)
    if not closest_title:
        print(f"No title found for \"{title}\" !!!")
        return

    # Get the ID and the rating percentage of the game we're trying to find recommendations for
    original_game_id = game_names.loc[game_names["name"] == closest_title]["appid"].values[0]
    original_rating_perc = all_ratings.loc[game_names["appid"] == original_game_id]["rating_percentage"].values[0]
    game_id_in_model = appid_to_index[original_game_id] # map it's actual game id to the model id
    
    # Obtain the result of the prediction - the num_recommendations closest games and their distances
    distances, indices = knn.kneighbors([perc_matrix[game_id_in_model]], n_neighbors=num_recommendations+1)
    distances, indices = distances[0], indices[0]
    combined = list(zip(distances, indices))[1:] # combine for easier access, and remove the game itself!!

    # Print recommendations
    print(f"======= GAME = {closest_title} =======")
    print(f"======= RATING PERCENTAGE = {percentage(original_rating_perc)} % =======\n")
    for distance, index in combined:
        current_game_id = index_to_appid[index]
        # The rating of the game corresponding to the above (real) id
        current_game_rating = all_ratings.loc[game_names["appid"] == current_game_id]["rating_percentage"].values[0]
        # Name of the game
        recommended_game = game_names.loc[game_names["appid"] == current_game_id]["name"].values[0] 
        print(f"\t{recommended_game} - {percentage(current_game_rating)} %")

# print(early[early["name"].str.contains(title)]["name"])
# text = joined_df.loc[joined_df["appid"] == 6270]["about_the_game"].values[0]

metrics = [
    COSINE,
    # EUCLIDEAN,
    # JACCARD,
]
num_recommendations = 20
description = False
games = [
    # "Team Fortress Classic",
    # "DOOM II",
    # "DmC: Devil May Cry",
    # "DiRT 4", 
    # "the witcher 2 assassins of kings",
    # "Middle-earth™: Shadow of War™",
    # "Among Us",
    "Trackmania United Forever",
    # "grid 2",
]
for metric in metrics:
    if description:
        fname = f"intermediate_data/{metric}_sim_description.pickle"
    else:
        fname = f"intermediate_data/{metric}_sim.pickle"
    mtx = read_or_calc_similarity(fname, metric)
    for game in games:
        content_based_similar_games(game, mtx, metric, num_recommendations)
        print()
        # collaborative_based_similar_games(game, num_recommendations)
        # print()

# df = pd.read_csv("sets/content_based_dataset.csv")
# df["about_the_game"] = df["about_the_game"].fillna("")