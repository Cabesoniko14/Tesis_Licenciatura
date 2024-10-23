from groq import Groq

client = Groq(
    api_key = "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"
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