# Голосовая генерация (TTS) — описание реализации

Этот документ описывает, как в проекте KidBot («Коровик») сделана генерация
голоса из текста. Цель — чтобы по этому описанию можно было собрать аналогичный
TTS в другом боте, в т.ч. с разными эмоциями и взрослым женским голосом.

---

## 1. Стек

- **Провайдер TTS:** Google Gemini (через официальный SDK `google-genai`).
- **Модель:** `gemini-2.5-flash-preview-tts`
  (константа `MODEL_TTS` в `config.py`).
- **Голос (prebuilt):** `Leda` — один из встроенных голосов Gemini
  (константа `TTS_VOICE`). Это женский, относительно молодой и «лёгкий» тембр.
  Полный список доступных голосов Gemini — Achernar, Achird, Algenib, Algieba,
  Alnilam, Aoede, Autonoe, Callirrhoe, Charon, Despina, Enceladus, Erinome,
  Fenrir, Gacrux, Iapetus, Kore, Laomedeia, Leda, Orus, Puck, Pulcherrima,
  Rasalgethi, Sadachbia, Sadaltager, Schedar, Sulafat, Umbriel, Vindemiatrix,
  Zephyr, Zubenelgenubi. Для «взрослого женского» обычно подходят
  **Kore, Charon (м), Aoede, Callirrhoe, Despina, Sulafat** — нужно слушать,
  у каждого свой характер. **Leda** звучит молодо/мягко, для взрослого скорее
  стоит попробовать **Kore** или **Sulafat**.
- **Конвертация аудио:** `ffmpeg` (внешний бинарь, должен быть в PATH).
  PCM → WAV (в памяти) → OGG/Opus.
- **Доставка в Telegram:** через `aiogram`, методом `reply_voice` с
  `BufferedInputFile(ogg_bytes, filename="voice.ogg")`.

---

## 2. Высокоуровневая логика

1. Пользователь получает обычный текстовый ответ от бота.
2. Под ответом прикреплена инлайн-кнопка «🔊 Озвучить» (`callback_data="tts"`).
3. При нажатии:
   - бот шлёт `chat_action: typing` (визуальная индикация «печатает»),
   - вызывает `text_to_speech(text)` — генерирует аудио,
   - отправляет ответом голосовое сообщение через `reply_voice`.
4. Если генерация не удалась — отправляет текстовое «не получилось».

Озвучка **по требованию**, а не автоматически — это экономит токены и даёт
пользователю выбор. Можно сделать иначе (см. п. 6 «варианты»).

---

## 3. Алгоритм `text_to_speech` (ядро)

```python
async def text_to_speech(text: str) -> bytes | None:
    resp = await ai.aio.models.generate_content(
        model=MODEL_TTS,
        contents=f"Прочитай вслух дружелюбным, весёлым тоном: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE,
                    )
                )
            ),
        ),
    )
    pcm_data = resp.candidates[0].content.parts[0].inline_data.data
    if not pcm_data:
        return None
    return await _pcm_to_ogg(pcm_data)
```

Ключевые моменты:

- **`response_modalities=["AUDIO"]`** — обязательно, без него модель вернёт текст.
- **`prebuilt_voice_config.voice_name`** — выбор тембра.
- **Инструкция по эмоции/стилю передаётся в самом `contents`**, как
  естественноязычный префикс перед текстом. Это и есть «эмоциональный
  стайл-промпт». Gemini TTS поддерживает такие управляющие фразы — модель
  слушается их по интонации.
  Пример: `"Прочитай вслух дружелюбным, весёлым тоном: {text}"`.
- На выходе — **сырые PCM-данные** (linear PCM, 16-бит, моно, 24 кГц).
  Их нельзя слать в Telegram напрямую: Telegram ждёт **OGG/Opus** для
  голосовых сообщений.

---

## 4. Конвертация PCM → OGG/Opus

```python
async def _pcm_to_ogg(pcm_data: bytes, sample_rate: int = 24000) -> bytes | None:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)     # mono
        wf.setsampwidth(2)     # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", "64k",
        "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ogg_data, err = await proc.communicate(wav_buf.getvalue())
    if proc.returncode != 0:
        return None
    return ogg_data
```

Шаги:

1. Заворачиваем PCM в WAV-контейнер в памяти (`io.BytesIO` + `wave`).
   Это нужно, чтобы у ffmpeg был валидный входной формат с заголовком
   (sample rate, channels, bit depth).
2. Запускаем `ffmpeg` через `asyncio.create_subprocess_exec` с пайпами:
   stdin ← WAV, stdout → OGG/Opus.
3. Кодек `libopus`, битрейт 64 кбит/с — хороший баланс качества/размера
   для голоса в Telegram.
4. Если ffmpeg вернул не 0 — пишем ошибку в лог и возвращаем `None`.

**Зависимость:** на машине должен быть установлен `ffmpeg`, и он должен
быть доступен в PATH. На Windows — поставить через `winget install ffmpeg`
или скачать с gyan.dev. В Docker — `apt install ffmpeg`.

---

## 5. Управление эмоциями — как расширить

Сейчас эмоция «зашита» одной строкой:
`"Прочитай вслух дружелюбным, весёлым тоном: {text}"`

Чтобы дать выбор эмоций, нужно:

### 5.1. Расширить сигнатуру

```python
EMOTION_PROMPTS = {
    "neutral":   "Прочитай спокойным, нейтральным тоном: {text}",
    "happy":     "Прочитай радостно и тепло, с улыбкой в голосе: {text}",
    "sad":       "Прочитай грустно, медленно, с сочувствием: {text}",
    "excited":   "Прочитай восторженно, энергично, с придыханием: {text}",
    "whisper":   "Прочитай шёпотом, тихо, доверительно: {text}",
    "angry":     "Прочитай раздражённо, резко, отрывисто: {text}",
    "seductive": "Прочитай мягко, бархатно, с лёгкой хрипотцой: {text}",
    "sarcastic": "Прочитай с явной иронией, растягивая слова: {text}",
}

async def text_to_speech(text: str, emotion: str = "neutral",
                        voice: str = "Kore") -> bytes | None:
    prompt = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["neutral"]).format(text=text)
    resp = await ai.aio.models.generate_content(
        model=MODEL_TTS,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )
    pcm_data = resp.candidates[0].content.parts[0].inline_data.data
    return await _pcm_to_ogg(pcm_data) if pcm_data else None
```

### 5.2. Как выбирать эмоцию

Варианты:
- **Ручной выбор пользователем** — несколько инлайн-кнопок:
  «🔊 Спокойно / 😊 Весело / 😢 Грустно / 🤫 Шёпотом».
  В `callback_data` кодировать эмоцию: `tts:happy`, `tts:sad`, и т.д.
- **Автоматический выбор моделью** — перед TTS прогонять текст через
  лёгкую модель (`gemini-2.5-flash-lite`) с промптом
  «определи эмоцию ответа: neutral/happy/sad/...». Получаешь лейбл,
  подставляешь в `EMOTION_PROMPTS`. Стоит токены — но «само играет».
- **Структурированный ответ от чат-модели** — попросить основную модель
  возвращать JSON `{"text": "...", "emotion": "..."}`. Тогда TTS уже
  знает, как читать.

### 5.3. Многоговорящий режим (диалог)

Gemini TTS также поддерживает **multi-speaker** через
`multi_speaker_voice_config` — можно задать двух спикеров с разными
голосами и читать диалог. Полезно, если бот «отыгрывает» сценки или
читает разговор персонажей. Структура:

```python
speech_config=types.SpeechConfig(
    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
        speaker_voice_configs=[
            types.SpeakerVoiceConfig(speaker="Анна", voice_config=...),
            types.SpeakerVoiceConfig(speaker="Борис", voice_config=...),
        ]
    )
)
```

И в `contents` пишется:
`"Анна: Привет!\nБорис: Здравствуй."`

---

## 6. Где это вкручено в бот (контекст)

- `brain.py` — функции `text_to_speech` и `_pcm_to_ogg` (ядро TTS).
- `main.py` — хендлер `handle_tts` (callback кнопки «Озвучить») и
  функция `_tts_kb()` (генерирует инлайн-клавиатуру с кнопкой).
- `config.py` — `MODEL_TTS`, `TTS_VOICE`.
- Кнопка прикрепляется к каждому текстовому ответу через `reply_markup=_tts_kb()`.

Поток данных:
```
[пользователь нажал кнопку]
  └─> handle_tts(callback)
       ├─> берёт text из callback.message.text
       ├─> brain.text_to_speech(text)
       │     ├─> Gemini API → PCM bytes
       │     └─> _pcm_to_ogg(pcm) → OGG/Opus bytes
       └─> callback.message.reply_voice(BufferedInputFile(ogg, "voice.ogg"))
```

---

## 7. Подводные камни / что важно

1. **PCM нужно обернуть в WAV перед ffmpeg.** Голый PCM ffmpeg примет только
   с явным указанием `-f s16le -ar 24000 -ac 1`, но через WAV-контейнер
   надёжнее и не зависит от точного знания формата.
2. **Sample rate Gemini TTS = 24000 Гц**, моно, 16-бит signed PCM.
   Если поменяется — звук поедет (ускорится / замедлится).
3. **Длина текста.** Telegram ограничивает голосовые сообщения по длине
   файла, но прежде — Gemini TTS имеет свой лимит на длину входа
   (несколько тысяч символов). Длинный текст лучше резать на куски и
   склеивать ffmpeg-ом через `concat`, либо просто отказываться от
   озвучки слишком длинных ответов.
4. **Стоимость.** TTS Gemini считается по символам входа, не по токенам.
   Для «весёлого ребёнка» это копейки, для длинных взрослых диалогов
   стоит посмотреть в pricing.
5. **Сеть.** `subprocess` под ffmpeg блокирует поток на время кодирования.
   Для коротких голосовых это ~100-300 мс, нормально. Для очень длинных
   стоит выносить в `asyncio.to_thread` или процесс-пул.
6. **Голос `Leda`** в текущем боте выбран как тёплый и подходящий ребёнку.
   Для взрослого женского — попробуй `Kore`, `Sulafat`, `Aoede`,
   `Despina`. У каждого свой характер, нужно слушать сэмплы.
7. **«Эмоциональный префикс» — это инструкция модели**, она не озвучивается.
   Модель понимает «Прочитай вслух весело: ...» как мета-команду и читает
   именно содержимое после двоеточия с заданной интонацией.
   Это работает на естественном языке — формулировки можно делать
   литературные: «как актриса в любовной сцене», «как уставшая учительница»,
   «шёпотом, словно открываешь секрет» — модель пытается это отыграть.

---

## 8. Минимальный шаблон под другой бот

```python
# requirements:
# google-genai
# aiogram (или твоя tg-библиотека)
# ffmpeg в PATH

import asyncio, io, wave
from google import genai
from google.genai import types

GEMINI_API_KEY = "..."
MODEL_TTS = "gemini-2.5-flash-preview-tts"

ai = genai.Client(api_key=GEMINI_API_KEY)

EMOTIONS = {
    "neutral":   "Прочитай спокойно и ровно",
    "happy":     "Прочитай радостно, с улыбкой в голосе",
    "sad":       "Прочитай грустно и медленно",
    "excited":   "Прочитай восторженно и энергично",
    "whisper":   "Прочитай шёпотом, доверительно",
    "seductive": "Прочитай мягко, бархатно",
}

async def tts(text: str, voice: str = "Kore", emotion: str = "neutral") -> bytes | None:
    style = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    resp = await ai.aio.models.generate_content(
        model=MODEL_TTS,
        contents=f"{style}: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    pcm = resp.candidates[0].content.parts[0].inline_data.data
    if not pcm:
        return None

    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "64k",
        "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ogg, _ = await proc.communicate(wav.getvalue())
    return ogg if proc.returncode == 0 else None
```

Вызов:
```python
audio = await tts("Привет, дорогой. Я скучала.", voice="Kore", emotion="seductive")
# дальше — отправляешь как voice message в telegram
```

---

## 9. Что стоит попробовать в новом боте

- **Сравнить голоса** — сгенерировать одну и ту же фразу разными
  `voice_name` и послушать. Сделать /demo команду, которая прогоняет
  список из 5-6 кандидатов.
- **Сравнить эмоции на одной фразе** — то же, но фиксированный голос,
  меняется `emotion`. Так быстро поймёшь, какие префиксы модель
  реально слышит, а какие игнорирует.
- **Скорость / темп** — добавить в префикс «не торопясь», «быстро»,
  «делая паузы между фразами». Gemini TTS отчасти управляется.
- **Хрипотца, шёпот, придыхание** — это работает: «с лёгкой хрипотцой»,
  «полушёпотом», «с придыханием». Хорошо звучит у Kore и Sulafat.
- **Не злоупотреблять эмодзи в тексте, который идёт в TTS.** Модель
  пытается их «прочитать» или игнорирует — лучше зачищать перед
  отправкой: `re.sub(r"[^\w\s.,!?-]", "", text)` или аккуратнее.

---

**TL;DR:**
Gemini TTS (`gemini-2.5-flash-preview-tts`) + выбор `voice_name` из
prebuilt-списка + натуральноязычный префикс с эмоцией в `contents` →
получаешь PCM 24 кГц моно → оборачиваешь в WAV → конвертируешь
ffmpeg-ом в OGG/Opus → шлёшь как voice в Telegram.
Эмоции управляются префиксом, голоса — параметром.
