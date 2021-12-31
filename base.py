import pandas as pd

from utils import COSINE, EUCLIDEAN, JACCARD
from utils import read_or_calc_similarity, train_model, find_closest_title, percentage

def content_based_similar_games(games, metric, num_recommendations=20, description=False):
    """
    For a given list of games, using the given metric,
    and the information whether to include the game description
    when finding recommendations, print out the wanted number of recommendations (default 20)
    """

    # Read necessary set
    df = pd.read_csv("sets/content_based_dataset.csv")
    df["about_the_game"] = df["about_the_game"].fillna("") # for some reason I need to do this
    temp = pd.DataFrame(df["name"], columns=["name"])

    # Read the similarity/distance matrix for this metric
    if description:
        filename = f"intermediate_data/{metric}_sim_description.pickle"
    else:
        filename = f"intermediate_data/{metric}_sim.pickle"
    matrix = read_or_calc_similarity(filename, metric)

    for game in games:
        # Get the most likely title for the current game
        closest_title = find_closest_title(temp, game)
        if not closest_title:
            print(f"No title found for \"{game}\" !!!!")
            continue

        print(f"======= GAME = {closest_title} =======")
        print(f"======= INCLUDE DESCRIPTION = {description} =======")
        print(f"======= METRIC = {metric} =======\n")

        indices = pd.Series(df.index, index=df["name"]) # for each game, get it's index
        index = indices[closest_title] # get the index for the title we're looking for
        similarity_scores = list(enumerate(matrix[index])) # get the similarity scores for this game with all others,
        # and enumerate them so we'll be able to detect which games it's closest to
        reverse = True
        if metric == EUCLIDEAN:
            reverse = False
        # Exclude the game itself and extract the required number of recommendations
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=reverse)[1:(num_recommendations+1)]

        game_indices = (i[0] for i in similarity_scores) # i = (index, score) so get just the indices
        # Time to retrieve and print out the recommendations
        for recommended_game in df["name"].iloc[game_indices].values:
            print(f"\t{recommended_game}")
        print()
        print()

def collaborative_based_similar_games(games, metric, num_recommendations=20):
    
    # Obtain the necessary data
    all_ratings = pd.read_csv("sets/collaborative_based_dataset.csv")
    game_names = pd.read_csv("sets/collaborative_based_dataset_names.csv")
    knn, perc_matrix, index_to_appid, appid_to_index = train_model(metric)

    for game in games:
        # Find the most likely title for this input title
        closest_title = find_closest_title(game_names, game)
        if not closest_title:
            print(f"No title found for \"{game}\" !!!")
            continue

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
        print(f"======= METRIC = {metric} =======")
        print(f"======= RATING PERCENTAGE = {percentage(original_rating_perc)} % =======\n")

        for distance, index in combined:
            current_game_id = index_to_appid[index]
            # The rating of the game corresponding to the above (real) id
            current_game_rating = all_ratings.loc[game_names["appid"] == current_game_id]["rating_percentage"].values[0]
            # Name of the game
            recommended_game = game_names.loc[game_names["appid"] == current_game_id]["name"].values[0] 
            print(f"\t{recommended_game} - {percentage(current_game_rating)} %")
        print()
        print()

metrics = [
    COSINE,
    EUCLIDEAN,
]
num_recommendations = 20
description = True
games = [
    # "Team Fortress Classic",
    # "DOOM II",
    # "DmC: Devil May Cry",
    # "dirt 4", 
    # "the witcher 2 assassins of kings",
    # "middle earth shadow of war",
    # "Among Us",
    "Trackmania United Forever",
    # "grid 2",
    # "dota 2",
    # "nba 2k19"
]
for metric in metrics:
    content_based_similar_games(games, metric, num_recommendations, description=description)
    # collaborative_based_similar_games(games, metric, num_recommendations)

# df = pd.read_csv("sets/content_based_dataset.csv")
# df["about_the_game"] = df["about_the_game"].fillna("")