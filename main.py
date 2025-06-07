from test.testing import output_label
from test.rewrite import polite_rewrite

import discord
import os
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# 機器人啟動時觸發
@bot.event
async def on_ready():
    print(f"機器人已登入：{bot.user}")


# 收到訊息時觸發（非指令）
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # 忽略自己

    result = output_label(message.content)
    if any(result[label]["pred"] for label in result):

        webhooks = await message.channel.webhooks()
        webhook = None
        for wh in webhooks:
            if wh.name == "Imitator":
                webhook = wh
                break
        if webhook is None:
            webhook = await message.channel.create_webhook(name="Imitator")

        await message.delete()  # 刪除有害訊息
        placeholder = await webhook.send(
            "改寫中...",
            username=message.author.display_name,
            avatar_url=message.author.avatar.url if message.author.avatar else None,
            wait=True,  # 記得要 wait=True 才能獲得訊息對象
        )
        rewritten = await polite_rewrite(message.content)
        await placeholder.edit(content=rewritten)

        try:
            await message.author.send(
                f"⚠️ 你剛才在 **#{message.channel.name}** 中的訊息包含有害內容，已被刪除。\n請遵守伺服器規則。"
            )
        except discord.Forbidden:
            print(f"❌ 無法私訊 {message.author}")


# 啟動 bot（請替換成你自己的 token）
bot.run(TOKEN)
