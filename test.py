from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

r = client.responses.create(
    model="gpt-4.1-mini",
    input="Responde solo con OK"
)

print(r.output_text)
