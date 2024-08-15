from pymilvus import Collection, db, connections

conn = connections.connect(host="10.66.12.37", port=19530)
db.using_database("text_video_db")
coll_name = 'text_video_vector'

collection = Collection(coll_name)

ids = [str(idx) for idx in range(10000)]
temp_str = ', '.join(ids)
query_expr = f'm_id in [{temp_str}]'
result = collection.delete(query_expr)