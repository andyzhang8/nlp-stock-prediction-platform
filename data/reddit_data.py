'''Fetches posts and comments from subreddits and saves them to a database'''

import requests

from db_helper import DB

database = DB(collection="reddit")


def fetch_and_save_posts(subreddit: str, limit=100):
    # Fetch posts from Reddit's JSON endpoint
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch posts. Status code: {response.status_code}")
        return

    posts = response.json().get("data", {}).get("children", [])
    print(f"Fetched {len(posts)} posts.")

    for post in posts:
        data = post['data']
        post_id = data['id']

        if data['num_comments'] > 5:  # Check for more than 5 comments
            title = data['title']
            content = data.get('selftext', '')
            
            # Fetch comments for the post
            comments_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            comments_response = requests.get(comments_url, headers=headers)

            if comments_response.status_code == 200:
                comments_data = comments_response.json()
                new_comments = [
                    comment['data']['body']
                    for comment in comments_data[1]['data']['children']
                    if 'body' in comment['data']
                ]
            else:
                new_comments = []

            # Check if the post already exists in the database
            existing_post = database.collection.find_one({"post_id": post_id})
            
            if existing_post:
                # If post exists, add new comments that are not already saved
                saved_comments = set(existing_post.get('comments', []))
                unique_new_comments = [comment for comment in new_comments if comment not in saved_comments]

                if unique_new_comments:
                    database.collection.update_one(
                        {"post_id": post_id},
                        {
                            "$set": {
                                "num_comments": data['num_comments'],
                                "content": content,
                            },
                            "$addToSet": {
                                "comments": {"$each": unique_new_comments}
                            }
                        }
                    )
                    print(f"Updated post: {title} with {len(unique_new_comments)} new comments.")
            else:
                # If post does not exist, insert a new entry
                post_data = {
                    "post_id": post_id,
                    "title": title,
                    "content": content,
                    "comments": new_comments,
                    "num_comments": data['num_comments'],
                    "created_utc": data['created_utc']
                }
                database.collection.insert_one(post_data)
                print(f"Saved new post: {title}")





if __name__ == '__main__':
    subreddits = ["pennystocks"]
    for subreddit in subreddits:
        fetch_and_save_posts(subreddit) 