from pymilvus import connections, db

conn = connections.connect(host="10.66.12.37", port=19530)
database = db.create_database("text_video_db")

db.using_database("text_video_db")
print(db.list_database())