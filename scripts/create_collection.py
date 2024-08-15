from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections


conn = connections.connect(host="10.66.12.37", port=19530)
db.using_database("text_video_db")

m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=512,)
video_id = FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=256,)
at_seconds = FieldSchema(name="at_seconds", dtype=DataType.INT32, max_length=256,)
schema = CollectionSchema(
  fields=[m_id, embeding, video_id, at_seconds],
  description="text to video embeding search",
  enable_dynamic_field=True
)

collection_name = "text_video_vector"
collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)