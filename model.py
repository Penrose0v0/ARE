import openai

from database import Database
from system_prompt import System_Prompt_Generator

class Model:
    def __init__(self, api_key, database: Database, system_prompt_generator: System_Prompt_Generator,
                 record_len=5, model="gpt-3.5-turbo", temperature=0, max_tokens=300,
                 top_p=1, frequency_penalty=0, presence_penalty=0):
        openai.api_key = api_key
        self.database = database
        self.system_prompt_generator = system_prompt_generator

        # Initiate the model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Initial personality and memory list
        self.messages = []
        self.record_list = []
        self.record_len = record_len
        self.personality = "You are a highly intelligent agent that can provide insightful responses and engage in " \
                           "meaningful conversations. I hope you can comply with the following instructions. Do not " \
                           "itemize or enumerate when you give responses. "

    def load_memory(self):
        with open('./memory.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            records = []
            for i in range(0, len(lines), 2):
                inp, out = lines[i], lines[i+1]
                record = (inp, out)
                records.append(record)
            self.database.insert_entities(records)

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_response(self, prompt):
        # Set system prompt
        system_prompt = self.personality
        generated_system_prompt = self.system_prompt_generator.get_system_prompt(self.record_list, prompt)
        system_prompt += generated_system_prompt
        print(generated_system_prompt)
        self.add_message("system", system_prompt)

        # Load record
        for ask, answer in self.record_list:
            self.add_message("user", ask)
            self.add_message("assistant", answer)

        # Search in memory DB with prompt and load relevant memory
        results = self.database.search(prompt)[::-1]
        uid_list = []
        distance_list = []
        for result in results:
            uid, user, assistant, distance = result
            self.add_message("user", user)
            self.add_message("assistant", assistant)
            uid_list.append(uid)
            distance_list.append(distance)

        # Reset memory retention and time of relevant memory
        self.database.review(uid_list)
        # for short in self.record_list:
        #     print([short])
        # self.database.show_all_data()

        # Update all memory and start to forget
        self.database.update_memory_retention(uid_list)
        self.database.forget()

        # Add current prompt
        self.add_message("user", prompt)

        # Get response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        response = response["choices"][0]["message"]["content"].strip()

        return response

    def save_memory(self):
        pass

    def run(self):
        self.load_memory()

        while True:
            prompt = input("U: ")
            if prompt == 'q':
                self.database.drop_collection()
                print('Over. ')
                break

            # Get the response from AI
            response = self.get_response(prompt)
            print("A: " + response)

            # Insert oldest short-term memory into memory DB if short-term memory is full
            current_dialog = (prompt + '\n', response + '\n')
            self.record_list.append(current_dialog)
            if len(self.record_list) > self.record_len:
                self.database.insert_entity(self.record_list.pop(0))

            print()

if __name__ == "__main__":
    with open("key.txt", 'r') as file:
        key = file.readline().strip()
