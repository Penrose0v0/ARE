from module import Module

class System_Prompt_Generator(Module):
    def __init__(self, chat_model="ft:gpt-3.5-turbo-0613:dng:system-prompt:7rOZ9ukc", temperature=0.5,
                 max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0):
        super().__init__(chat_model=chat_model, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                         frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        system_prompt = \
            "You are a conversation system-prompt generator. " \
            "You need to analyze user input, then provide system prompt without giving any answer directly " \
            "in order to give instructions and align tasks to the question-answering GPT. " \
            "The conversation should guide the AI's responses in a way that maintains context and coherence. " \
            "In the conversation, 'U' means 'user' while 'A' means 'assistant'. \n"

        samples = [
            (
                "U: Hi there, bot! I'm feeling a bit lost in the world of programming. Any advice on how to stay "
                "motivated and keep learning?\nA: Hey! It's completely normal to feel overwhelmed sometimes. One "
                "great way to stay motivated is by setting achievable goals. Start with small coding projects that "
                "interest you and gradually challenge yourself with more complex tasks.\n",

                "U: Thanks for the advice! I'm currently learning Python, but I often get stuck with bugs and "
                "errors.\n ",

                "Provide advice and tips for someone learning programming, "
                "particularly focusing on staying motivated and dealing with bugs and errors "
                "while learning Python. "
            ),

            (
                "U:Do you know who is Donald Trump? \nA: Yes, I am familiar with Donald Trump. He is a prominent figure"
                "in both business and politics. Donald Trump is a businessman and television personality who served as "
                "the 45th President of the United States from January 2017 to January 2021. He was known for his real "
                "estate ventures, including the Trump Organization, before entering politics. During his presidency, "
                "Trump was known for his unconventional communication style, use of social media, and controversial "
                "policy decisions.\n",

                "U: Can you try to speak a paragraph in Trump's tone. \n",

                "Generate a paragraph of text in a tone reminiscent of Donald Trump's speaking style. "
                "Write as if you were him, using his distinct mannerisms and language."
            ),

            (
                "U: Do you know who is Donald Trump?\n"
                "A: Yes, I am familiar with Donald Trump. Donald Trump is a businessman and politician who served as "
                "the 45th President of the United States from 2017 to 2021. He was known for his background in real "
                "estate and his role as a television personality on the show 'The Apprentice' before entering "
                "politics. His presidency was marked by controversial policies, a focus on immigration and trade, "
                "and a unique communication style.\n",

                "U: Can you try to learn a little about Trump's speech, like speak in his tone. ",

                "Generate a response in which the AI mimics Donald Trump's speech style. Use his distinct language and "
                "mannerisms to create a paragraph of text that resembles the way he communicates."
            )
        ]

        self.prefix1 = "What can the system prompt be to respond to the following input: \n(\n{})"
        self.prefix2 = "This is conversation history record, which you can use as reference: \n(\n{})\n"
        self.fix_response = "I will refer to the conversation record you provided me and find the relevant content " \
                            "in the record in the next round of conversation and ignore the irrelevant content"

        self.samples = [{"role": "system", "content": system_prompt}]
        for sample in samples:
            messages = [
                {"role": "user",
                 "content": self.prefix2.format(sample[0])},
                {"role": "assistant",
                 "content": self.fix_response},
                {"role": "user",
                 "content": self.prefix1.format(sample[1])},
                {"role": "assistant",
                 "content": sample[2]}
            ]
            for message in messages:
                self.samples.append(message)

    def get_system_prompt(self, record_list, user_input):
        # Get current conversation
        history_record = ""
        if len(record_list):
            for record in record_list:
                text = "U: " + record[0] + "A: " + record[1]
                history_record += text
            history_record = self.prefix2.format(history_record)
        current_conversation = self.prefix1.format("U: " + user_input)
        prompt = history_record + current_conversation

        # Load samples and conversation
        messages = []
        for sample in self.samples:
            messages.append(sample)
        messages.append({"role": "user", "content": prompt})

        # Get response
        response = self.get_response(messages)
        return response


class Temperature_Generator(Module):
    def __init__(self, temperature=0.2, max_tokens=8, top_p=1, frequency_penalty=0, presence_penalty=0):
        super().__init__(temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                         frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        system_prompt = \
            "You are a temperature parameter generator. " \
            "You need to generate appropriate temperature parameter for the question-answering GPT based on its " \
            "system prompt and user input without giving any answer directly. " \
            "Your output should be a number. "

        samples = [
            (
                "Hello. \n",

                "Engage in a friendly conversation with the user. Respond to their greeting and initiate a polite and "
                "welcoming interaction. \n",

                "0.7"
            ),

            (
                "Do you know what is the human life expectancy in the US? \n",

                "This is a question about a specific statistic. The AI should provide the current average life "
                "expectancy for humans in the United States. \n",

                "0.3"
            )
        ]

        self.prefix1 = "User input: {}"
        self.prefix2 = "System prompt: {}"

        self.samples = [{"role": "system", "content": system_prompt}]
        for sample in samples:
            messages = [
                {"role": "user",
                 "content": self.prefix1.format(sample[0]) + self.prefix2.format(sample[1])},
                {"role": "assistant",
                 "content": sample[2]},
            ]
            for message in messages:
                self.samples.append(message)

    def get_temperature(self, generated_system_prompt, user_input):
        # Load samples and prompt
        messages = []
        for sample in self.samples:
            messages.append(sample)
        prompt = self.prefix1.format(user_input) + self.prefix2.format(generated_system_prompt)
        messages.append({"role": "user", "content": prompt})

        # Get response
        while True:
            response = self.get_response(messages)
            try:
                return eval(response)
            except:
                pass


if __name__ == "__main__":
    system_prompt_generator = System_Prompt_Generator()
    temperature_generator = Temperature_Generator()

    for system_prompt_sample in system_prompt_generator.samples:
        print(system_prompt_sample)
    print()
    for temperature_sample in temperature_generator.samples:
        print(temperature_sample)
    print()

    tmp_system_prompt = "Generate a response in which the AI mimics Donald Trump's speech style. Use his distinct " \
                        "language and mannerisms to create a paragraph of text that resembles the way he communicates."
    tmp_user_input = "Can you try to learn a little about Trump's speech, like speak in his tone."
    tmp_temperature = temperature_generator.get_temperature(tmp_system_prompt, tmp_user_input)
    print(tmp_temperature)
