from groq import Groq

client = Groq(
    api_key = ""
)

chat_completion = client.chat.completions.create(
    messages = [
        {
            "role": "user",
            "content": "Explain to me what Reinforcement Learning is"
        }
    ],
    model = "llama3-8b-8192"
)

print(chat_completion.choices[0].message.content)