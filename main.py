import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
from io import BytesIO
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from scipy.spatial.distance import euclidean
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Инициализация бота
API_TOKEN = 'YOUR_TELEGRAM_API_TOKEN'  # Замените на ваш токен
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Установка токена Hugging Face для pyannote.audio
os.environ["PYANNOTE_AUDIO_AUTH"] = "YOUR_HUGGINGFACE_TOKEN"  # Замените на ваш токен

# Загрузка предобученной модели диаризации
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)

# Словарь для хранения пользователей
users = {}

# Кнопки
main_menu = ReplyKeyboardMarkup(resize_keyboard=True)
main_menu.add(KeyboardButton("Принять образец голоса"))
main_menu.add(KeyboardButton("Загрузить аудиофайл совещания"))
main_menu.add(KeyboardButton("Получить протокол совещания"))

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply("Добро пожаловать! Выберите действие.", reply_markup=main_menu)

# Обработчик кнопки "Принять образец голоса"
@dp.message_handler(lambda message: message.text == "Принять образец голоса")
async def voice_sample(message: types.Message):
    await message.reply("Введите вашу фамилию и имя.")
    await bot.register_next_step_handler(message, save_user_data)

async def save_user_data(message: types.Message):
    full_name = message.text.strip()
    users[message.from_user.id] = {'name': full_name}
    await message.reply(f"Спасибо, {full_name}. Теперь отправьте голосовое сообщение длительностью около 10 секунд.")
    await bot.register_next_step_handler(message, save_voice_sample)

async def save_voice_sample(message: types.Message):
    if message.voice:
        file_info = await bot.get_file(message.voice.file_id)
        voice_file = await bot.download_file(file_info.file_path)

        # Преобразуем голосовое сообщение в формат wav для дальнейшей обработки
        audio = AudioSegment.from_file(BytesIO(voice_file.read()), format="ogg")
        wav_data = BytesIO()
        audio.export(wav_data, format="wav")

        # Извлекаем характеристики голоса (MFCC)
        voice_features = extract_voice_features(wav_data)
        users[message.from_user.id]['voice_features'] = voice_features

        await message.reply("Голосовой образец сохранен!")
    else:
        await message.reply("Пожалуйста, отправьте голосовое сообщение.")

# Обработчик загрузки аудиофайла совещания
@dp.message_handler(lambda message: message.text == "Загрузить аудиофайл совещания")
async def upload_meeting_audio(message: types.Message):
    await message.reply("Пожалуйста, загрузите аудиофайл совещания в формате .wav.")
    await bot.register_next_step_handler(message, process_meeting_audio)

# Глобальная переменная для хранения последнего аудиофайла совещания
meeting_audio_data = None

async def process_meeting_audio(message: types.Message):
    global meeting_audio_data
    if message.audio or message.document:
        file_info = await bot.get_file(message.audio.file_id if message.audio else message.document.file_id)
        audio_file = await bot.download_file(file_info.file_path)

        # Проверяем формат файла и при необходимости конвертируем в WAV
        if message.audio and message.audio.mime_type != 'audio/wav':
            audio = AudioSegment.from_file(BytesIO(audio_file.read()))
            wav_data = BytesIO()
            audio.export(wav_data, format="wav")
            meeting_audio_data = wav_data
        else:
            meeting_audio_data = BytesIO(audio_file.read())

        await message.reply("Аудиофайл совещания загружен!")
    else:
        await message.reply("Пожалуйста, загрузите аудиофайл.")

# Обработчик кнопки "Получить протокол совещания"
@dp.message_handler(lambda message: message.text == "Получить протокол совещания")
async def get_meeting_protocol(message: types.Message):
    global meeting_audio_data
    if meeting_audio_data is None:
        await message.reply("Сначала загрузите аудиофайл совещания.")
        return

    # Выполняем диаризацию аудио
    segments = diarize_audio(meeting_audio_data)

    # Распознаем речь и сопоставляем говорящих
    transcription, recognized_speakers = transcribe_and_identify_speakers(meeting_audio_data, segments, users)

    # Генерируем протокол
    protocol = generate_meeting_protocol(transcription, recognized_speakers, segments)
    await message.reply(f"Протокол совещания:\n{protocol}")

# Извлечение голосовых характеристик (например, MFCC)
def extract_voice_features(wav_data):
    y, sr = librosa.load(wav_data, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Сравнение голосов на основе евклидова расстояния
def compare_voices(sample_features, user_features):
    return euclidean(sample_features, user_features)

# Диаризация аудиофайла
def diarize_audio(audio_data):
    # Сохраняем аудио во временный файл
    temp_audio_file = 'temp_meeting_audio.wav'
    with open(temp_audio_file, 'wb') as f:
        f.write(audio_data.getbuffer())

    # Выполняем диаризацию
    diarization = pipeline(temp_audio_file)

    # Удаляем временный файл
    os.remove(temp_audio_file)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_info = {
            'speaker_id': speaker,
            'start': turn.start,
            'end': turn.end,
        }
        segments.append(segment_info)
    return segments

# Распознавание речи и идентификация говорящих
def transcribe_and_identify_speakers(audio_data, segments, users):
    recognizer = sr.Recognizer()
    transcription = []
    recognized_speakers = []

    # Загружаем полный аудиофайл
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)

    # Для каждого сегмента выполняем распознавание и идентификацию говорящего
    for segment in segments:
        start = segment['start']
        end = segment['end']
        duration = end - start

        # Извлекаем сегмент аудио
        with sr.AudioFile(audio_data) as source:
            source.audio_reader.seek(int(start * source.SAMPLE_RATE))
            segment_audio = source.audio_reader.read(int(duration * source.SAMPLE_RATE))

            # Создаем AudioData для распознавания
            segment_audio_data = sr.AudioData(segment_audio, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

        # Распознаем речь в сегменте
        try:
            text = recognizer.recognize_google(segment_audio_data, language="ru-RU")
        except sr.UnknownValueError:
            text = "[Неразборчиво]"
        except sr.RequestError as e:
            text = f"[Ошибка распознавания: {e}]"

        transcription.append(text)

        # Извлекаем характеристики голоса сегмента
        segment_audio_bytes = audio_data_slice(audio_data, start, end)
        segment_features = extract_voice_features(segment_audio_bytes)

        # Сопоставляем голос с пользователями
        min_distance = float('inf')
        recognized_user = "Неизвестный говорящий"

        for user_id, user_data in users.items():
            user_features = user_data['voice_features']
            distance = compare_voices(segment_features, user_features)

            if distance < min_distance:
                min_distance = distance
                recognized_user = user_data['name']

        recognized_speakers.append(recognized_user)

    return transcription, recognized_speakers

# Функция для извлечения части аудио по времени
def audio_data_slice(audio_data, start_time, end_time):
    audio = AudioSegment.from_file(audio_data)
    segment = audio[start_time * 1000:end_time * 1000]  # Конвертируем в миллисекунды
    segment_data = BytesIO()
    segment.export(segment_data, format="wav")
    segment_data.seek(0)
    return segment_data

# Генерация протокола совещания с именами и таймстемпами
def generate_meeting_protocol(transcription, recognized_speakers, segments):
    protocol = ""
    for i, (speaker, phrase) in enumerate(zip(recognized_speakers, transcription)):
        start_time = segments[i]['start']
        timestamp = format_timestamp(start_time)
        protocol += f"{timestamp} - {speaker}: {phrase}\n"
    return protocol

# Форматирование таймстемпа
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# Запуск бота
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    executor.start_polling(dp, skip_updates=True)
