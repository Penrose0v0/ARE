import openai

from model import Model
from database import Database
from system_prompt import System_Prompt_Generator

if __name__ == "__main__":
    with open("key.txt", 'r') as file:
        key = file.readline().strip()
    database = Database(api_key=key, collection_name='memory', similarity_threshold=0.36, forgetting_threshold=0.65)
    system_prompt_generator = System_Prompt_Generator(api_key=key)
    model = Model(api_key=key, database=database, system_prompt_generator=system_prompt_generator, record_len=3)
    model.run()
