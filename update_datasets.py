import pandas as pd
from requests import get
from fuzzywuzzy import fuzz
from utils import percentage
import codecs
import json

"""
Some functions to update datasets are found here.
"""

def steamspy_tag_data_mapping():
    """
    NOTE: NOT PERFECT, YOU HAVE TO GO THROUGH mapping.json ON YOUR OWN TO
    FIX MISTAKES BUT IT'S WAY FASTER THAN JUST DOING IT BY HAND

    Defines a mapping between the "bad" steamspy tags (ones that are
    all lowercase, with underscores, etc) and the good ones (ones that
    are uppercase, no underscores, -, etc)

    So an example mapping is "warhammer_40k": "Warhammer 40K"
    or "turn_based_combat": "Turn-Based Combat" etc

    NOTE #2: It is enough to run this once and to have a quick look 
    at mapping.json to fix errors
    """
    main_df = pd.read_csv("sets/steam.csv")
    tags_df = pd.read_csv("sets/steamspy_tag_data.csv")

    # Extract all the good tags
    good_tags = main_df["steamspy_tags"].values
    good_tags_set = set()
    for tags in good_tags:
        for tag in tags.split(";"):
            good_tags_set.add(tag)

    # Get the bad ones
    bad_tags = tags_df.columns[1:].values # Remove the appid column, we don't need it
    
    # For a given bad tag, return the closest match out of all the good tags
    def closest_tag(bad_tag):
        matches = []
        for good_tag in good_tags_set:
            ratio = fuzz.ratio(bad_tag.lower(), good_tag.lower())
            if ratio >= 60:
                matches.append((good_tag, ratio))
        if not matches:
            return None
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        return matches[0][0]

    # Define the mapping and save
    mapping = {}
    for bad_tag in bad_tags:
        mapping[bad_tag] = closest_tag(bad_tag) # can be null which is fine, we fix by hand

    with open("mapping.json", "w") as f:
        json.dump(mapping, f)

def steamspy_tag_data_rename_cols():
    """
    Update the steamspy_tag_data.csv file to have column names from mapping.json
    """
    df = pd.read_csv("sets/steamspy_tag_data.csv")
    with open("mapping.json", "r") as f:
        mapping = json.load(f)
        df = df.rename(mapping, axis="columns")
        df.to_csv("sets/steamspy_tag_data_updated.csv", index=False)

################################################################################################

def update_ratings():
    """
    For all the games in the steam.csv dataset, retrieve their
    newest amount of positive and negative ratings.
    """
    # base link: https://store.steampowered.com/appreviews/<appid>?json=1&language=all

    df = pd.read_csv("sets/steam.csv")
    ids = df["appid"].values # get all ids
    num_to_update = len(ids)
    ids_skipped = []
    for i, game_id in enumerate(ids):
        # Retrieve result for this game
        link = f"https://store.steampowered.com/appreviews/{game_id}?json=1&language=all"
        response = get(link)
        # On the fly fixing, initially it was just result = response.json()
        try:
            result = response.json()
        except:
            try:
                decoded_data = codecs.decode(response.text.encode(), 'utf-8-sig') # https://www.howtosolutions.net/2019/04/python-fixing-unexpected-utf-8-bom-error-when-loading-json-data/
                result = json.loads(decoded_data)
            except:
                print(f"GAME ID {game_id} SKIPPED !!!!")
                ids_skipped.append(game_id)
                continue
        
        query_summary = result.get("query_summary")
        num_reviews = query_summary.get("num_reviews")
        if num_reviews == 0:
            # since this game does not exist (probably won't happen but just as a safety measure)
            continue

        positive = query_summary.get("total_positive") # number of total positive reviews
        negative = query_summary.get("total_negative") # number of total negative reviews
        df.loc[df["appid"] == game_id, "positive_ratings"] = positive # update
        df.loc[df["appid"] == game_id, "negative_ratings"] = negative
        print(f"Reviews, Game id {game_id}, remaining: {num_to_update - (i + 1)} out of {num_to_update}")

    print(f"Skipped {len(ids_skipped)} games")
    print(ids_skipped)
    df.to_csv("sets/steam_updated.csv", index=False) # save to separate dataset

def add_more_tags():
    """
    In steam.csv, there are only 3 steam spy tags associated with every game.
    The goal is to add two more.
    """
    main = pd.read_csv("sets/steam_updated.csv") # the update_ratings function has to be ran first
    tags = pd.read_csv("sets/steamspy_tag_data_updated.csv")
    main_ids = main["appid"].values
    num_to_update = len(main_ids)

    for i, game_id in enumerate(main_ids):
        row = tags.loc[tags["appid"] == game_id] # get the corresponding row from the tags dataset
        row = row[row.columns[1:]] # remove appid column
        row_index = row.index.tolist()[0] # get the index of the row
        row = row.sort_values(by=row_index, axis=1, ascending=False) # sort in descending order
        first_five = row.iloc[:, :5] # extract the first 5 columns and values (5 closest tags for that game)
        first_five_tags = ";".join(first_five.columns.tolist()) # get just the column names (tags), appended together
        
        main.loc[main["appid"] == game_id, "steamspy_tags"] = first_five_tags # update
        print(f"Tags, Game id {game_id}, remaining: {num_to_update - (i + 1)} out of {num_to_update}")

    main.to_csv("sets/steam_updated.csv", index=False) # save to the updated dataset

def remove_edition_from_games():
    """
    Remove any 'edition' from the name of the title
    Also it's not perfect (it doesn't remove it for every 
    single game) but it works well enough
    """

    with open("editions.txt", "r") as f:
        editions = [line[:-1] for line in f.readlines()] # [:-1] to remove the \n

    df = pd.read_csv("sets/steam_updated.csv")
    ids = df["appid"].values
    num_to_update = len(ids)
    for i, game_id in enumerate(ids):
        game = df[df["appid"] == game_id]["name"].values[0] # get the current name
        for edition in editions:
            if edition in game:
                game = game.replace(edition, "") # remove any Edition from the game name
                break
        df.loc[df["appid"] == game_id, "name"] = game # update (or it may stay the same)
        print(f"Editions, Game id {game_id}, remaining: {num_to_update - (i + 1)} out of {num_to_update}")

    df.to_csv("sets/steam_updated.csv", index=False)

# import time
# start = time.perf_counter()
# update_ratings()
# add_more_tags()
# remove_edition_from_games()
# print(time.perf_counter() - start)

# df = pd.read_csv("sets/steam_updated.csv")
# editions = df[df["name"].str.contains("[^exp]edition", case=False)]["name"].values
# for edition in editions:
#     print(edition)
# print(len(editions))
