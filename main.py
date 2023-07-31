import openai
# import redis
from database import Database

key = "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"

class Model:
    def __init__(self, api_key, memory: Database, human_name='Q', ai_name='A',
                 short_term_len=5, model="text-davinci-003", temperature=0, max_tokens=300,
                 top_p=1, frequency_penalty=0, presence_penalty=0):
        openai.api_key = api_key
        self.human_name = human_name
        self.ai_name = ai_name

        # Initiate the model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Initial personality and memory list
        self.memory = memory
        self.short_term_list = []
        self.short_term_len = short_term_len
        self.personality = ''

    def set_personality_and_memory(self):
        # Load personality
        with open('./personality.txt', 'r', encoding='utf-8') as file:
            self.personality += file.readline() + '\n'

        # Load memory
        with open('./memory.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            dialogues = []
            for i in range(0, len(lines), 2):
                inp, out = lines[i], lines[i+1]
                dialogue = self.human_name + ': ' + inp + self.ai_name + ': ' + out
                dialogues.append(dialogue)
            self.memory.insert_entities(dialogues)

    def get_response(self, prompt):
        # Add personality and memory to the prompt
        new_prompt = self.personality
        for dialogue in self.short_term_list:
            new_prompt += dialogue

        # Search in memory DB with prompt
        results = self.memory.search(prompt)[::-1]
        uid_list = []
        distance_list = []
        for result in results:
            uid, dialogue, distance = result
            new_prompt += dialogue
            uid_list.append(uid)
            distance_list.append(distance)

        # Reset memory retention and time of relevant memory
        self.memory.review(uid_list)
        # for short in self.short_term_list:
        #     print([short])
        # self.memory.show_all_data()

        # Update all memory and start to forget
        self.memory.update_memory_retention(uid_list)
        self.memory.forget()

        # Add current prompt
        new_prompt += self.human_name + ': ' + prompt + '\n' + self.ai_name + ': '
        # print([new_prompt])

        # Get response
        response = openai.Completion.create(
            model=self.model,
            prompt=new_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        response = response["choices"][0]["text"].strip()

        return response

    def run(self):
        self.set_personality_and_memory()

        while True:
            prompt = input(self.human_name + ': ')
            if prompt == 'q':
                self.memory.drop_collection()
                print('Over. ')
                break

            # Get the response from AI
            response = self.get_response(prompt)
            print(self.ai_name + ': ' + response)

            # Insert oldest short-term memory into memory DB if short-term memory is full
            current_dialog = self.human_name + ': ' + prompt + '\n' + self.ai_name + ': ' + response + '\n'
            self.short_term_list.append(current_dialog)
            if len(self.short_term_list) > self.short_term_len:
                self.memory.insert_entity(self.short_term_list.pop(0))

database = Database(api_key=key, collection_name='memory', similarity_threshold=0.36, forgetting_threshold=0.1)
test = Model(api_key=key, memory=database)
test.run()
