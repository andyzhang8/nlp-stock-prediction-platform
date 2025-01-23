from pymongo import MongoClient

class DB:
    def __init__(self, collection="", db_name="stocks"):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.collection = self.db[collection]

        print(f"Database collection {collection} initialized")

    