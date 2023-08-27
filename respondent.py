from module import Module

class Respondent(Module):
    def __init__(self, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0):
        super().__init__(max_tokens=max_tokens, top_p=top_p,
                         frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

        self.root_system_prompt = \
            "You are a highly intelligent agent that can provide insightful responses and " \
            "engage in meaningful conversations. " \
            "You will speak freely like a human, rather than being rigid like a robot. " \
            "I hope you can comply with the following instructions. " \
            "Give me a few sentences instead of an article, which means summarize your answer as much as possible. " \
            "Never itemize or enumerate like '1... 2... 3...' when you give responses! " \
            "When I ask you something, prioritize finding answers from our conversation records, " \
            "which means you should recall and summarize the information that is relevant to current dialogue. "

        self.system_prompt = ''

    @staticmethod
    def make_messages(dialogue):
        return {"role": "user", "content": dialogue[0]}, {"role": "assistant", "content": dialogue[1]}

    def set_system_prompt(self, generated_system_prompt):
        self.system_prompt += self.root_system_prompt + generated_system_prompt

    def set_temperature(self, generated_temperature):
        self.temperature = generated_temperature

    def get_final_response(self, data_list, record_list, user_input):
        # Load system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        self.system_prompt = ''

        # Load data
        for data in data_list:
            user, assistant = self.make_messages(data)
            messages.append(user)
            messages.append(assistant)

        # Load record
        for record in record_list:
            user, assistant = self.make_messages(record)
            messages.append(user)
            messages.append(assistant)

        # Load user input
        messages.append({"role": "user", "content": user_input})

        # Get response
        response = self.get_response(messages)
        return response

if __name__ == "__main__":
    pass
