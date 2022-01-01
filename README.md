# Game recommendation engine

A project for the subject "Strojno učenje". A few simple recommender systems for video games are implemented.

# Data sets

`steam.csv`, `steams_description_data.csv` and `steamspy_tag_data.csv` are the original and unedited data sets downloaded from [here](https://www.kaggle.com/nikdavis/steam-store-games).

`steam_updated.csv` is the updated version of the original `steam.csv` data set. The updates are:

-   Each game's positive and negative rating count is fresh, rather than from 2019. In `update_datasets.py`, have a look at `update_ratings`.
-   The edition of each game is removed (i.e. games do not contain Special Edition, Collectors Edition, Definitive Edition, ...). These editions are listed in the `editions.txt` file, the corresponding function is `remove_edition_from_games` in `update_datasets.py`.
-   Two more steam spy tags (i.e. genres) have been included for each game. The function is `add_more_tags` in `update_datasets.py`.
-   Since `steam_updated.csv` already exists, i.e. since all of these updates have been performed, the only thing that may need to be ran is the `update_ratings` function, to update the positive and negative ratings for each game. This will take a few hours to complete, and there will be (with the current implementation) around 200 or so skipped games.

`steamspy_tag_data_updated.csv` is more or less the same data set as `steamspy_tag_data.csv`, however the column names have been updated to be more readable. Have a look at `mapping.json`, as well as the functions `steamspy_tag_data_mapping` and `steamspy_tag_data_rename_cols` in `update_datasets.py`.

`content_based_dataset.csv`, and `collaborative_based_dataset.csv` are data sets used directly for implementing. The functions `make_content_based_dataset` and `make_collaborative_based_dataset` in `utils.py` are in charge for making them. Be sure to invoke them with the parameter `steam_updated` in order to actually account for the aforementioned updates to the data set.

# Similarity matrices

In order to run the script, the four 5.25 GB large distance/similarity matrices must be made. Obviously, I was unable to push these.

To make these matrices, it is enough to open `utils.py`, add the following code:

```python
df = pd.read_csv("sets/content_based_dataset.csv")
df["about_the_game"] = df["about_the_game"].fillna("")
vectorize_and_similaritize(df, [COSINE, EUCLIDEAN])
```

and run it. Be sure to make and enter a virtual environment before that, and install all the packages from `requirements.txt`. Calculating the matrices will take some time and will slow down your computer. If, after making one matrix, you get an error saying that numpy cannot allocate 5.25GB of memory, or something along those lines, the idea is to create one matrix at a time, i.e. to run the program 4 times. To do this, in `vectorize_and_similaritize`, comment out a part of the code intended for making one of the distance matrices, and then run the function for example, like this: `vectorize_and_similaritize(df, [EUCLIDEAN])`.

When the two Euclidean matrices have been made, just repeat the process but instead of `EUCLIDEAN`, use `COSINE`.

# How to run

Open `base.py`, choose your metrics, number of recommendations, whether to include descriptions in the recommendations, and your games. Then, just run the script with `python base.py`. Make sure you're in a virtual environment with the installed packages from `requirements.txt`.
