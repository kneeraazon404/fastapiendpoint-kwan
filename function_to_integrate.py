import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json

uri = "mongodb+srv://Datainput:inputdata@cluster0.jpw0qjv.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client['cmedai']  # replace with your database name

# Access the collection
collection = db['教科书']  # replace with your collection name


def autocomplete_search(query, collection, path="病名", index="diseaseName"):
    pipeline = [
        {"$search": {
            "index": index, 
            "autocomplete": {
                "query": query, 
                "path": path
            }
        }},
        {"$limit": 1},
        {"$project": {"_id": 0, "病名": 1, "预防与调摄": 1}},  # include both the "病名" and "预防与调摄" fields in the returned documents
    ]
    result = collection.aggregate(pipeline)
    return list(result)


result = autocomplete_search("胃", collection)

for doc in result:
    print(f"{doc['病名']}")
    print("预防与调摄:")
    for item in doc['预防与调摄']:
        print(f"  - {item}")
    print()
