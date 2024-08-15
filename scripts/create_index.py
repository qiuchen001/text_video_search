from pymilvus import Collection, utility, connections, db

conn = connections.connect(host="10.66.12.37", port=19530)
db.using_database("text_video_db")

index_params = {
  "metric_type": "IP",
  "index_type": "IVF_FLAT",
  "params": {"nlist": 1024}
}

collection = Collection("text_video_vector")
collection.create_index(
  field_name="embeding",
  index_params=index_params
)

utility.index_building_progress("text_video_vector")