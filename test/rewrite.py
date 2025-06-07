import cohere
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("COHERE")

# 初始化 Cohere，用你自己的 API 金鑰
co = cohere.Client(TOKEN)  # ✅ 請替換為你的 Cohere API 金鑰


async def polite_rewrite(sentence):
    prompt = f"""Please rewrite the following sentence to make it more polite, keeping the original meaning, and respond with only one single sentence:

    Original: {sentence}
    Polite:"""

    response = co.chat(message=prompt, temperature=0.3)

    return response.text.strip()


# 範例句子
if __name__ == "__main__":
    original = "You're ugly."
    polite_version = polite_rewrite(original)
    print("委婉改寫:", polite_version)
