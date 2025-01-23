import data
from data.reddit_data import fetch_and_save_posts


subreddits = ["pennystocks"]

for subreddit in subreddits:
    fetch_and_save_posts(subreddit)
