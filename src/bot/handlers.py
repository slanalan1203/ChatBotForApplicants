import asyncio

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from src.rag.pipeline import answer

router = Router()

GREETING = (
    "Привет! Я бот для абитуриентов ВШЭ. Задавай вопросы про поступление: "
    "сроки, документы, программы, стипендии, общежитие. "
    "Я автоматический помощник, важную информацию уточняй у приёмной комиссии."
)

@router.message(Command("start"))
async def start(msg: Message):
    await msg.answer(GREETING)

@router.message(Command("help"))
async def help_(msg: Message):
    await msg.answer(GREETING)

@router.message(F.text)
async def on_text(msg: Message):
    result = await asyncio.to_thread(answer, msg.text)
    text = result["answer"]
    seen = set()
    unique = []
    for s in result["sources"]:
        key = s["url"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
        if len(unique) == 3:
            break
    if unique and "в моих источниках" not in text.lower():
        srcs = "\n".join(f"• {s['title']}: {s['url']}" for s in unique)
        text = f"{text}\n\nИсточники:\n{srcs}"
    await msg.answer(text)
