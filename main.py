import time

from respondent import Respondent
from database import Database
from generator import System_Prompt_Generator, Temperature_Generator

class Model:
    def __init__(self, respondent: Respondent, database: Database,
                 system_prompt_generator: System_Prompt_Generator, temperature_generator: Temperature_Generator, ):
        # Initialize modules
        self.respondent = respondent
        self.database = database
        self.system_prompt_generator = system_prompt_generator
        self.temperature_generator = temperature_generator

        self.data_list = []
        self.record_list = []
        self.system_prompt = ''
        self.temperature = -1

    def load_data(self):
        with open('memory/memory.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            records = []
            for i in range(0, len(lines), 2):
                inp, out = lines[i], lines[i + 1]
                record = (inp, out)
                records.append(record)
            self.database.insert_entities(records)

    def get_response(self, user_input: str, view_status=False):
        start = time.time()

        # Get data_list
        results = self.database.search(user_input)
        uid_list = []
        for result in results:
            uid, user, assistant, _ = result
            self.data_list.append((user, assistant))
            uid_list.append(uid)
        self.database.review(uid_list)
        self.database.update_memory_retention(uid_list)
        self.database.forget()

        # Get system prompt based on record_list, user_input
        self.system_prompt = self.system_prompt_generator.get_system_prompt(self.record_list, user_input)
        self.respondent.set_system_prompt(self.system_prompt)

        # Get temperature based on system_prompt, user_input
        self.temperature = self.temperature_generator.get_temperature(self.system_prompt, user_input)
        self.respondent.set_temperature(self.temperature)

        # Get response
        assistant_output = self.respondent.get_final_response(self.data_list, self.record_list, user_input)

        end = time.time()

        # View status
        if view_status:
            print(f"\t-System Prompt: {self.system_prompt.strip()}")
            print(f"\t-Temperature: {self.temperature}")
            print(f"\t-Time: {end - start}")

        return assistant_output

    def update_record_list(self, user_input, assistant_output):
        record = (user_input, assistant_output)
        self.record_list.append(record)
        if len(self.record_list) > 5:
            self.database.insert_entity(self.record_list.pop(0))

    def clear_data_list(self):
        self.data_list = []

    def initialize_system_prompt_and_temperature(self):
        self.system_prompt = ''
        self.temperature = -1


if __name__ == "__main__":
    fmt = "\n-----{:^20}-----"

    print(fmt.format("Initializing"))
    model = Model(Respondent(), Database(), System_Prompt_Generator(), Temperature_Generator())
    model.load_data()

    print(fmt.format("Dialogue beginning"))
    while True:
        prompt = input("U: ")
        if prompt == 'q':
            print(fmt.format("Dialogue ending"))
            model.database.drop_collection()
            break
        prompt += '\n'

        response = model.get_response(prompt, view_status=True)
        print("A: " + response.strip())

        model.update_record_list(prompt, response)
        model.clear_data_list()
        model.initialize_system_prompt_and_temperature()
        print()
    print(fmt.format("Process finished"))
