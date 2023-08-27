import openai

with open("key/key.txt", 'r') as file:
    key = file.readline().strip()

class Module:
    def __init__(self, api_key=key, chat_model="gpt-3.5-turbo", embed_model="text-embedding-ada-002",
                 temperature=0, max_tokens=300, top_p=1, frequency_penalty=0, presence_penalty=0):
        openai.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def get_response(self, messages):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        response = response["choices"][0]["message"]["content"].strip() + '\n'
        return response

    def get_embedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model=self.embed_model
        )
        return response["data"][0]["embedding"]

if __name__ == "__main__":
    test = Module()
    messages = [
        {
            "role": "user",
            "content": "Do you know what is the life expectancy in the US? "
        },

        {
            "role": "user",
            "content": "Do you know what happened in Spain in 1992? "
        }
    ]
    answer = test.get_response(messages)
    print(answer)
