import openai
import time
from math import exp
import uuid
import redis
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

# Get current rate of memory using forgetting curve
def forgetting_curve(x, a, b=0.082, k=0.5):
    # x -> time, a -> times of revision, b -> related to strength of memory, k -> related to effect of revision
    c = b * exp(-k * a)
    return exp(-c * x)

class Database:
    def __init__(self, api_key, collection_name,
                 similarity_threshold=0.36, search_limit=5, forgetting_threshold=0.1,
                 max_length=300, dim=1536, scan_count=100):
        openai.api_key = api_key
        self.collection_name = collection_name
        self.scan_count = scan_count
        self.similarity_threshold = similarity_threshold
        self.search_limit = search_limit
        self.forgetting_threshold = forgetting_threshold

        # Connect to Redis
        print("Start connecting to Redis")
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
        self.redis_client.flushdb()

        # Connect to Milvus
        print("Start connecting to Milvus")
        connections.connect("default", host="localhost", port="19530")

        # Create collection
        has = utility.has_collection(collection_name)
        print(f"Does collection '{collection_name}' exist in Milvus: {has}")

        if not has:
            fields = [
                FieldSchema(name="uid", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="user", dtype=DataType.VARCHAR, max_length=max_length),
                FieldSchema(name="assistant", dtype=DataType.VARCHAR, max_length=max_length),
                FieldSchema(name="dialogue", dtype=DataType.VARCHAR, max_length=max_length),
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

        else:
            self.collection = Collection(collection_name)

        # Load collection
        print("Start loading")
        self.collection.load()

    # Insert single entity
    def insert_entity(self, record):
        # Insert into Milvus
        text = record[0] + record[1]
        uid = uuid.uuid4().int >> 65
        entity = [[uid], [record[0]], [record[1]], [text], [get_embedding(text)]]
        self.collection.insert(entity)

        # Insert into Redis
        self.redis_client.hset(uid, 'memory_retention', 1)
        self.redis_client.hset(uid, 'time', 0)
        self.redis_client.hset(uid, 'revision', 0)

    # Insert multiple entities
    def insert_entities(self, record_list):
        # Insert into Milvus
        uid_list, user_list, assistant_list, dialogue_list, embedding_list = [], [], [], [], []
        for record in record_list:
            text = record[0] + record[1]
            uid_list.append(uuid.uuid4().int >> 65)
            user_list.append(record[0])
            assistant_list.append(record[1])
            dialogue_list.append(text)
            embedding_list.append(get_embedding(text))
        entities = [uid_list, user_list, assistant_list, dialogue_list, embedding_list]
        self.collection.insert(entities)

        # Insert into Redis
        pipe = self.redis_client.pipeline()
        for uid in uid_list:
            pipe.hset(uid, 'memory_retention', 1)
            pipe.hset(uid, 'time', 0)
            pipe.hset(uid, 'revision', 0)
        pipe.execute()

    # Search from Milvus
    def search(self, text):
        vector_to_search = [get_embedding(text)]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = self.collection.search(data=vector_to_search, anns_field="embedding", param=search_params,
                                        limit=self.search_limit, output_fields=["uid", "user", "assistant"],
                                        expr=None, consistency_level="Strong")
        # print(results)
        filtered_results = [result for result in results[0] if result.distance < self.similarity_threshold]

        result_list = []
        for hit in filtered_results:
            result_list.append([hit.entity.get('uid'), hit.entity.get('user'),
                                hit.entity.get('assistant'), hit.distance])

        return result_list

    # Review and update in Redis
    def review(self, uid_list):
        pipe = self.redis_client.pipeline()
        for uid in uid_list:
            revision = int(self.redis_client.hget(uid, 'revision'))
            pipe.hset(uid, 'memory_retention', 1)
            pipe.hset(uid, 'time', 0)
            pipe.hset(uid, 'revision', revision + 1)
        pipe.execute()

    # Update memory retention
    def update_memory_retention(self, protect_list):
        cursor = 0
        pipe = self.redis_client.pipeline()
        while True:
            cursor, uid_list = self.redis_client.scan(cursor=cursor, count=self.scan_count)
            for uid in uid_list:
                if uid not in protect_list:
                    # Get current data
                    current = self.redis_client.hgetall(uid)
                    current_time = int(current['time'])
                    current_revision = int(current['revision'])

                    # Start to update
                    new_memory_retention = forgetting_curve(x=current_time, a=current_revision)
                    pipe.hset(uid, 'memory_retention', new_memory_retention)
                    pipe.hset(uid, 'time', current_time + 1)
                    pipe.hset(uid, 'revision', current_revision)

            pipe.execute()
            if cursor == 0:
                break

    # Delete entities whose memory retention is less than forgetting threshold
    def forget(self):
        cursor = 0
        pipe = self.redis_client.pipeline()
        while True:
            cursor, uid_list = self.redis_client.scan(cursor=cursor, count=self.scan_count)
            for uid in uid_list:
                current_memory_retention = eval(self.redis_client.hget(uid, 'memory_retention'))
                if current_memory_retention <= self.forgetting_threshold:
                    pipe.delete(uid)
                    self.collection.delete(expr=f'uid in [{uid}]')
            pipe.execute()
            if cursor == 0:
                break

    # Show all key-value in the database
    def show_all_data(self):
        uid_list = self.redis_client.keys('*')
        for uid in uid_list:
            print(uid, self.redis_client.hgetall(uid))

    # Drop collection
    def drop_collection(self):
        print(f"Drop collection '{self.collection_name}'")
        print("Flush all keys from Redis")
        utility.drop_collection(self.collection_name)
        self.redis_client.flushdb()

if __name__ == '__main__':
    with open("key.txt", 'r') as file:
        key = file.readline().strip()
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
