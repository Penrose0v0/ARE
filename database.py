import openai
import uuid
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

class Database:
    def __init__(self, api_key, collection_name, dim=1536):
        openai.api_key = api_key
        self.collection_name = collection_name

        # Connection
        print("Start connecting to Milvus")
        connections.connect("default", host="localhost", port="19530")

        # Create collection
        has = utility.has_collection(collection_name)
        print(f"Does collection '{collection_name}' exist in Milvus: {has}")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="dialogue", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields)

        print(f"Create collection '{collection_name}'")
        self.collection = Collection(collection_name, schema)

        # Create index
        print("Start creating index IVF_FLAT")
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.collection.create_index(field_name="embedding", index_params=index)

        # Load collection
        print("Start loading")
        self.collection.load()

    # Insert single entity
    def insert_entity(self, text):
        entity = [[uuid.uuid4().int >> 65], [text], [get_embedding(text)]]
        self.collection.insert(entity)

    # Insert multiple entities
    def insert_entities(self, text_list):
        id_list, dialogue_list, embedding_list = [], [], []
        for text in text_list:
            id_list.append(uuid.uuid4().int >> 65)
            dialogue_list.append(text)
            embedding_list.append(get_embedding(text))
        entities = [id_list, dialogue_list, embedding_list]
        self.collection.insert(entities)

    # Search
    def search(self, text, limit=5):
        vector_to_search = [get_embedding(text)]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        result = self.collection.search(vector_to_search, "embedding", search_params,
                                        limit=limit, output_fields=["dialogue"], consistency_level="Strong")
        result_list = []
        for hits in result:
            for hit in hits:
                result_list.append([hit.distance, hit.entity.get('dialogue')])

        return result_list

    # Drop collection
    def drop_collection(self):
        print(f"Drop collection '{self.collection_name}'")
        utility.drop_collection(self.collection_name)

if __name__ == '__main__':
    key = "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"
    database = Database(key, 'test')

    human_name = "Q"
    ai_name = "A"
    with open('./memory.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            inp, out = lines[i], lines[i + 1]
            dialogue = human_name + ": " + inp + ai_name + ": " + out
            print(dialogue)
            database.insert_entity(dialogue)
