import openai
# import redis
from database import Database

key = "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"

class Model:
    def __init__(self, api_key, memory, human_name='Q', ai_name='A',
                 model="text-davinci-003", temperature=0, max_tokens=300,
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

        # Search in memory DB with prompt
        results = self.memory.search(prompt, limit=1)
        for result in results:
            _, dialogue = result
            new_prompt += dialogue

        # Add current prompt
        new_prompt += self.human_name + ': ' + prompt + '\n' + self.ai_name + ': '
        # print("This is a test: ", [new_prompt])

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

            # Add the dialogue to memory
            current_dialog = self.human_name + ': ' + prompt + '\n' + self.ai_name + ': ' + response + '\n'
            self.memory.insert_entity(current_dialog)

database = Database(key, 'memory')
test = Model(key, database)
test.run()
