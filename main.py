import openai
# import redis

key = "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"

class Model:
    def __init__(self, api_key, human_name='Q', ai_name='A',
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
        self.all_memory_list = []  # str(input + '\n' + output) for each element
        self.long_memory_list = []  # (input, output) for each element
        self.short_memory_list = []  # [input, output, Forgetting_value] for each element
        self.forgotten_list = []  # Ready to forget
        self.personality = ''

    def set_personality_and_memory(self):
        self.personality += "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\n"

        # The followings are examples
        self.short_memory_list.append(['What is human life expectancy in the United States?\n', 'Human life expectancy in the United States is 78 years.\n'])
        self.short_memory_list.append(['Where were the 1992 Olympics held?\n', 'The 1992 Olympics were held in Barcelona, Spain.\n'])
        self.short_memory_list.append(['How many squigs are in a bonk?\n', 'Unknown\n'])

    def get_response(self, prompt):
        # Add personality and memory to the prompt
        new_prompt = self.personality
        for short_memory in self.short_memory_list:
            inp, out = short_memory
            new_prompt += self.human_name + ': ' + inp
            new_prompt += self.ai_name + ': ' + out
            new_prompt += '\n'

        # Add the current prompt
        new_prompt += self.human_name + ': ' + prompt + '\n' + self.ai_name + ': '

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
                print('Over. ')
                break

            # Get the response from AI
            response = self.get_response(prompt)
            print(self.ai_name + ': ' + response)

            # Add the dialog to memory
            current_dialog = [prompt + '\n', response + '\n']
            self.short_memory_list.append(current_dialog)

test = Model(key)
test.run()
