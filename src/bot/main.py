import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from src.bot.handlers import router

async def main():
    load_dotenv()
    bot = Bot(token=os.environ["TELEGRAM_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
