from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import numpy as np
import librosa
import pyworld as pw
import matplotlib
matplotlib.use('Agg')  # Безголовый режим для matplotlib
import matplotlib.pyplot as plt
from funasr import AutoModel
import pypinyin
import unicodedata
import base64
import re
import random
from io import BytesIO
import soundfile as sf
import logging
import traceback

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hskk_coach.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HSKK Coach API", version="1.0.0")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка ASR model (paraformer-zh-streaming с VAD для точной сегментации)
asr_model = None
try:
    logger.info("Загрузка ASR модели...")
    asr_model = AutoModel(
        model="paraformer-zh-streaming", 
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 60000},
        disable_update=True,
        device="cpu"  # Явно указываем CPU для стабильности
    )
    logger.info("ASR модель успешно загружена")
        
except Exception as e:
    logger.error(f"Ошибка загрузки ASR модели: {e}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    logger.info("Будет использован fallback режим без распознавания речи")
    asr_model = None

def normalize_text(text):
    # Убрать пробелы, пунктуацию и нормализовать
    text = re.sub(r"[^\w]", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text

def prepare_audio_for_asr(audio_path):
    """Подготавливает аудиофайл для ASR модели (конвертирует WebM в WAV)"""
    try:
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Файл не найден: {audio_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("Файл пустой")
        
        logger.info(f"Подготовка аудио: {audio_path} ({file_size} байт)")
        
        # Проверяем заголовок файла для определения формата
        with open(audio_path, 'rb') as f:
            header = f.read(20)
            logger.info(f"Заголовок файла (hex): {header.hex()}")
            
            # Проверяем форматы
            if header.startswith(b'RIFF') and b'WAVE' in header:
                logger.info("Формат: WAV")
                real_format = 'wav'
            elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xfa'):
                logger.info("Формат: MP3")
                real_format = 'mp3'
            elif header.startswith(b'\x1aE\xdf\xa3'):  # WebM magic number
                logger.info("Формат: WebM")
                real_format = 'webm'
            else:
                logger.warning(f"Неизвестный формат, пробуем обработать. Заголовок: {header[:20].hex()}")
                real_format = 'unknown'
        
        y = None
        sr = None
        
        # Если это WebM, конвертируем через ffmpeg
        if real_format == 'webm':
            logger.info("Конвертация WebM в WAV через ffmpeg")
            try:
                import subprocess
                import tempfile as tf
                
                # Определяем путь к ffmpeg
                ffmpeg_path = fr"ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
                
                # Проверяем существование ffmpeg
                if not os.path.exists(ffmpeg_path):
                    # Пробуем альтернативные пути
                    alt_paths = [
                        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                        r"C:\ffmpeg\bin\ffmpeg.exe",
                        "ffmpeg.exe"  # если в PATH
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            ffmpeg_path = alt_path
                            break
                    else:
                        raise Exception(f"ffmpeg не найден. Проверьте пути: {ffmpeg_path}")
                
                # Создаем временный wav файл в папке temp
                temp_wav_path = os.path.join("temp", f"converted_{os.path.basename(audio_path)}.wav")
                try:
                    # Конвертируем WebM в WAV через ffmpeg
                    cmd = [
                        ffmpeg_path, '-i', audio_path, '-ar', '16000', '-ac', '1', 
                        '-c:a', 'pcm_s16le', '-y', temp_wav_path
                    ]
                    logger.info(f"Запуск ffmpeg: {' '.join(cmd)}")
                    
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
                    logger.info(f"WebM успешно конвертирован в WAV")
                    logger.debug(f"ffmpeg stderr: {result.stderr}")
                    
                    # Проверяем созданный файл
                    if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 44:
                        logger.info(f"Конвертированный WAV файл: {os.path.getsize(temp_wav_path)} байт")
                        return temp_wav_path
                    else:
                        raise Exception("Конвертированный файл пустой или некорректный")
                        
                except subprocess.CalledProcessError as ffmpeg_error:
                    logger.error(f"Ошибка ffmpeg: {ffmpeg_error}")
                    logger.error(f"stderr: {ffmpeg_error.stderr}")
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                    raise
                except FileNotFoundError:
                    logger.error(f"ffmpeg не найден по пути: {ffmpeg_path}")
                    raise Exception(f"ffmpeg не найден по пути: {ffmpeg_path}")
                    
            except Exception as webm_error:
                logger.error(f"Ошибка конвертации WebM: {webm_error}")
                raise
        
        # Для WAV и MP3 используем librosa
        logger.info("Загрузка аудио через librosa")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            logger.info(f"Аудио успешно загружено: длина={len(y)}, sample_rate={sr}")
        except Exception as librosa_error:
            logger.error(f"Ошибка librosa: {librosa_error}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise Exception(f"Не удалось загрузить аудиофайл: {librosa_error}")
        
        if y is None or len(y) == 0:
            raise Exception("Не удалось загрузить аудиоданные")
        
        # Создаем временный wav файл в папке temp
        temp_wav_path = os.path.join("temp", f"processed_{os.path.basename(audio_path)}.wav")
        
        try:
            # Используем wave для создания WAV файла
            import wave
            
            # Конвертируем в 16-bit PCM
            if y.dtype != np.int16:
                y_int16 = (y * 32767).astype(np.int16)
            else:
                y_int16 = y
            
            with wave.open(temp_wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # моно
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sr)
                wav_file.writeframes(y_int16.tobytes())
            
            logger.info(f"Аудио сохранено как WAV: {temp_wav_path}")
            
            # Проверяем что файл создался корректно
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 44:
                logger.info(f"WAV файл создан: {os.path.getsize(temp_wav_path)} байт")
                return temp_wav_path
            else:
                raise Exception(f"WAV файл некорректен: размер={os.path.getsize(temp_wav_path) if os.path.exists(temp_wav_path) else 0}")
                
        except Exception as wave_error:
            logger.error(f"Ошибка wave: {wave_error}")
            
            # Fallback на soundfile
            try:
                sf.write(temp_wav_path, y, sr, subtype='PCM_16')
                logger.info(f"Аудио сохранено через soundfile: {temp_wav_path}")
                return temp_wav_path
            except Exception as sf_error:
                logger.error(f"Ошибка soundfile: {sf_error}")
                raise Exception(f"Не удалось сохранить WAV: {wave_error}")
        
    except Exception as e:
        logger.error(f"Ошибка подготовки аудио: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def extract_tones_from_pinyin(pinyin_str):
    tones = []
    for word in pinyin_str.split():
        match = re.search(r"[a-z]+([1-5])", word)
        if match:
            tone = int(match.group(1))
            # Преобразуем 5 тон в 0 для нейтрального
            tones.append(0 if tone == 5 else tone)
        else:
            tones.append(0)
    return tones

def analyze_pitch_contour(pitch_segment, global_mean, global_std):
    """Анализ контура высоты тона для определения тона"""
    mean_pitch = np.mean(pitch_segment)
    std_pitch = np.std(pitch_segment)
    
    # Относительная высота
    relative_height = (mean_pitch - global_mean) / global_std if global_std > 0 else 0
    
    # Анализ контура
    start_val = pitch_segment[0]
    end_val = pitch_segment[-1]
    mid_val = pitch_segment[len(pitch_segment) // 2]
    
    # Наклон контура
    if len(pitch_segment) > 1:
        slope = (end_val - start_val) / len(pitch_segment)
    else:
        slope = 0
    
    # Классификация тона
    if std_pitch < 10:  # плоский контур
        if relative_height > 0.5:
            return 1  # высокий плоский
        elif relative_height < -0.5:
            return 3  # низкий плоский
        else:
            return 0  # нейтральный
    elif slope > 0.1:  # поднимающийся
        return 2
    elif slope < -0.1:  # падающий
        return 4
    else:  # сложный контур
        if mid_val < start_val * 0.9 and end_val > mid_val * 1.1:
            return 3  # V-образный (тон 3)
        else:
            return 0

def classify_tones_from_f0(f0, sr, text):
    # Простая и надежная классификация тонов на основе F0 контура
    tones = []
    syllables = text

    # Фильтруем нулевые значения
    f0_valid = f0[f0 > 0]
    if len(f0_valid) == 0:
        return [0] * len(syllables)
    
    # Глобальные статистики
    global_mean = np.mean(f0_valid)
    global_std = np.std(f0_valid)
    
    # Разделяем F0 на сегменты по слогам
    num_syllables = len(syllables)
    segment_len = len(f0) // num_syllables if num_syllables > 0 else 1

    for i in range(num_syllables):
        start = i * segment_len
        end = (i + 1) * segment_len
        if end > len(f0):
            end = len(f0)
        
        pitch_segment = f0[start:end]
        
        if len(pitch_segment) == 0:
            tones.append(0)
            continue

        # Фильтруем нулевые значения
        valid_pitch = pitch_segment[pitch_segment > 0]
        if len(valid_pitch) < 2:
            tones.append(0)
            continue

        # Используем ту же логику анализа
        tone = analyze_pitch_contour(valid_pitch, global_mean, global_std)
        tones.append(tone)
    
    return tones

def analyze_audio_real(audio_path, reference_text, reference_pinyin):
    """Реальный анализ аудио с определением тонов"""
    try:
        # Нормализуем текст
        reference_text = normalize_text(reference_text)
        
        # Получаем пиньинь из текста
        ref_pinyin_tone3_list = pypinyin.pinyin(reference_text, style=pypinyin.TONE3)
        ref_pinyin = " ".join(["".join(word) for word in ref_pinyin_tone3_list])
        ref_tones = extract_tones_from_pinyin(ref_pinyin)

        # Распознаем речь из аудио
        user_text = ""
        
        if asr_model:
            try:
                # Подготавливаем аудиофайл для ASR
                prepared_audio_path = prepare_audio_for_asr(audio_path)
                is_temp_file = prepared_audio_path != audio_path
                
                logger.info(f"Распознавание речи из файла: {prepared_audio_path}")
                logger.info(f"Файл существует: {os.path.exists(prepared_audio_path)}")
                logger.info(f"Размер файла: {os.path.getsize(prepared_audio_path) if os.path.exists(prepared_audio_path) else 'N/A'} байт")
                logger.info(f"Абсолютный путь: {os.path.abspath(prepared_audio_path)}")
                
                # Пробуем распознать речь
                try:
                    res = asr_model.generate(input=prepared_audio_path)
                    logger.info("ASR распознавание успешно")
                    
                    if res and len(res) > 0:
                        user_text = normalize_text(res[0]["text"])
                        logger.info(f"Распознанный текст: {user_text}")
                    else:
                        user_text = reference_text  # Fallback
                        logger.info("ASR вернул пустой результат, используем эталонный текст")
                        
                except Exception as asr_error:
                    logger.error(f"ASR ошибка: {asr_error}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    
                    # Дополнительная диагностика
                    logger.info("Пробуем альтернативные методы...")
                    
                    # Метод 1: Используем numpy массив
                    try:
                        y, sr = librosa.load(prepared_audio_path, sr=16000)
                        res = asr_model.generate(input=y)
                        if res and len(res) > 0:
                            user_text = normalize_text(res[0]["text"])
                            logger.info(f"Распознанный текст (метод 1): {user_text}")
                        else:
                            raise Exception("Пустой результат")
                    except Exception as e1:
                        logger.error(f"Метод 1 не сработал: {e1}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        
                        # Метод 2: Используем бинарные данные
                        try:
                            with open(prepared_audio_path, 'rb') as f:
                                audio_data = f.read()
                            res = asr_model.generate(input=audio_data)
                            if res and len(res) > 0:
                                user_text = normalize_text(res[0]["text"])
                                logger.info(f"Распознанный текст (метод 2): {user_text}")
                            else:
                                raise Exception("Пустой результат")
                        except Exception as e2:
                            logger.error(f"Метод 2 не сработал: {e2}")
                            logger.error(f"Stack trace: {traceback.format_exc()}")
                            user_text = reference_text
                            logger.info("Используем эталонный текст (все методы не сработали)")
                
                # НЕ удаляем временный файл здесь - удалим в конце
                # if is_temp_file and os.path.exists(prepared_audio_path):
                #     os.unlink(prepared_audio_path)
                #     logger.info(f"Временный файл удален: {prepared_audio_path}")
                logger.info(f"Временный файл будет удален в конце: {prepared_audio_path}")
                    
            except Exception as e:
                logger.error(f"Общая ASR ошибка: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                user_text = reference_text  # Fallback
                logger.info("Используем эталонный текст из-за ошибки ASR")
        else:
            user_text = reference_text  # Fallback если модель не загружена
            logger.info("ASR модель не загружена, используем эталонный текст")
        
        logger.info(f"Анализ аудио файла: {audio_path}")
        logger.info(f"Эталонный текст: {reference_text}")
        logger.info(f"Эталонный пиньинь: {reference_pinyin}")

        # Извлекаем F0 и классифицируем тоны - используем сконвертированный файл
        final_audio_path = prepared_audio_path if 'prepared_audio_path' in locals() else audio_path
        logger.info(f"Используем файл для F0 анализа: {final_audio_path}")
        y, sr = librosa.load(final_audio_path, sr=16000, dtype=np.float64)
        
        # Используем librosa для извлечения F0 (pyworld может быть недоступен)
        try:
            logger.info("Извлечение F0 с помощью PyWORLD...")
            # Пробуем разные методы PyWORLD
            if hasattr(pw, 'dio'):
                _f0, t = pw.dio(y, sr)
                f0 = pw.stonemask(y, _f0, t, sr)
            elif hasattr(pw, 'harvest'):
                _f0, t = pw.harvest(y, sr)
                f0 = pw.stonemask(y, _f0, t, sr)
            else:
                raise Exception("PyWORLD функции не найдены")
            logger.info("F0 успешно извлечен с помощью PyWORLD")
        except Exception as e:
            logger.error(f"PyWORLD ошибка: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            logger.info("Используем librosa для извлечения F0...")
            try:
                # Fallback на librosa
                f0, t = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                # Берем среднюю частоту для каждого временного шага
                f0 = np.mean(f0, axis=0)
                t = librosa.times_like(f0, sr=sr)
                logger.info("F0 успешно извлечен с помощью librosa")
            except Exception as e2:
                logger.error(f"librosa тоже не сработал: {e2}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                # Создаем пустые данные для F0
                f0 = np.zeros(100)
                t = np.linspace(0, len(y)/sr, 100)
                logger.info("Используем пустые F0 данные")
        
        user_tones = classify_tones_from_f0(f0, sr, reference_text)

        # Создаем график
        plot_path = create_f0_plot(f0, t, reference_text, ref_tones, user_tones)

        # Сравниваем тона и создаем детальный анализ
        syllable_analysis = []
        correct_count = 0
        
        user_pinyin_tone3_list = pypinyin.pinyin(user_text, style=pypinyin.TONE3)
        
        for i, (user_tone, ref_tone) in enumerate(zip(user_tones, ref_tones)):
            char_ref = reference_text[i] if i < len(reference_text) else ''
            
            # Получаем пиньинь для пользователя
            if i < len(user_pinyin_tone3_list):
                pinyin_user_base = "".join(user_pinyin_tone3_list[i])
                pinyin_user_with_tone = pinyin_user_base[:-1] + str(user_tone) if pinyin_user_base[-1].isdigit() else pinyin_user_base + str(user_tone)
            else:
                pinyin_user_with_tone = f"{char_ref}({user_tone})"
            
            # Получаем эталонный пиньинь
            if i < len(ref_pinyin_tone3_list):
                pinyin_ref_base = "".join(ref_pinyin_tone3_list[i])
                pinyin_ref = pinyin_ref_base[:-1] + str(ref_tone) if pinyin_ref_base[-1].isdigit() else pinyin_ref_base + str(ref_tone)
            else:
                pinyin_ref = f"{char_ref}({ref_tone})"
            
            is_correct = user_tone == ref_tone
            if is_correct:
                correct_count += 1
            
            syllable_analysis.append({
                'user_syllable': pinyin_user_with_tone,
                'correct_syllable': pinyin_ref,
                'correct': is_correct,
                'character': char_ref
            })

        # Вычисляем точность
        accuracy = round((correct_count / len(ref_tones)) * 100) if ref_tones else 0
        
        # Создаем данные для графика
        chart_data = {
            'labels': [f'Слог {i+1}' for i in range(len(syllable_analysis))],
            'values': [100 if item['correct'] else 30 for item in syllable_analysis]
        }

        result = {
            'recognized_text': user_text,
            'target_text': reference_text,
            'accuracy': accuracy,
            'syllables': syllable_analysis,
            'chart_data': chart_data,
            'f0_plot': plot_path,
            'user_tones': user_tones,
            'reference_tones': ref_tones,
            'temp_audio_path': final_audio_path  # Добавляем путь к временному файлу
        }
        
        logger.info(f"Анализ завершен. Точность: {accuracy}%")
        return result

    except Exception as e:
        logger.error(f"Ошибка анализа аудио: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # Возвращаем результат с ошибкой
        return {
            'error': f'Ошибка анализа: {str(e)}',
            'recognized_text': reference_text,
            'target_text': reference_text,
            'accuracy': 0,
            'syllables': [],
            'chart_data': {'labels': [], 'values': []}
        }

def create_f0_plot(f0, t, reference_text, ref_tones, user_tones):
    """Создает график F0 и возвращает base64 изображение"""
    try:
        plt.figure(figsize=(12, 4))
        
        # График F0 пользователя
        plt.plot(t, f0, 'b-', label='Пользователь F0', alpha=0.7)
        
        # Добавляем сегменты тонов
        if len(reference_text) > 0:
            num_syllables = len(reference_text)
            segment_len = len(t) // num_syllables if num_syllables > 0 else 1
            
            colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'blue', 4: 'orange'}
            tone_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
            
            for i in range(min(num_syllables, len(user_tones))):
                start_time = i * segment_len / len(t) * t[-1]
                end_time = (i + 1) * segment_len / len(t) * t[-1]
                tone = user_tones[i] if i < len(user_tones) else 0
                char = reference_text[i] if i < len(reference_text) else ''
                
                plt.axvspan(start_time, end_time, alpha=0.2, color=colors.get(tone, 'gray'))
                plt.text((start_time + end_time) / 2, max(f0) * 0.9, f'{char}\n{tone_names.get(tone, tone)}', 
                        ha='center', va='center', fontsize=8)
        
        plt.xlabel("Time (s)")
        plt.ylabel("F0 (Hz)")
        plt.title("Анализ тонов пользователя")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сохраняем в base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Ошибка создания графика: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл index.html не найден")

@app.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    text: str = Form(...),
    pinyin: str = Form(...)
):
    """Анализ аудио с определением тонов"""
    
    # Проверяем формат файла - поддерживаем WebM, WAV, MP3
    allowed_types = ['audio/webm', 'audio/wav', 'audio/wave', 'audio/x-wav', 'audio/mp3', 'audio/mpeg']
    if audio.content_type and audio.content_type not in allowed_types:
        logger.warning(f"Неподдерживаемый тип файла: {audio.content_type}, пробуем обработать")
    
    # Создаем папку temp если она не существует
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Сохраняем аудио в папку temp
    import uuid
    unique_filename = f"{uuid.uuid4().hex[:8]}_{audio.filename}"
    temp_audio_path = os.path.join(temp_dir, unique_filename)
    
    # Читаем содержимое файла
    content = await audio.read()
    logger.info(f"Получен файл: {audio.filename}, размер: {len(content)} байт")
    logger.info(f"Content-type: {audio.content_type}")
    
    # Сохраняем файл
    with open(temp_audio_path, "wb") as temp_audio:
        temp_audio.write(content)
    
    logger.info(f"Файл сохранен: {temp_audio_path}")
    
    # Проверяем файл после сохранения
    if not os.path.exists(temp_audio_path):
        raise HTTPException(status_code=500, detail="Файл не был сохранен")
    
    saved_size = os.path.getsize(temp_audio_path)
    if saved_size == 0:
        raise HTTPException(status_code=400, detail="Сохраненный файл пустой")
    
    logger.info(f"Файл успешно сохранен: {saved_size} байт")
    
    try:
        # Анализируем произношение
        result = analyze_audio_real(temp_audio_path, text, pinyin)
        
        if 'error' in result:
            logger.error(f"Ошибка в результате анализа: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Сохраняем путь к временному файлу для удаления
        temp_wav_path = result.get('temp_audio_path')
        
        # Удаляем temp_audio_path из результата перед отправкой клиенту
        if 'temp_audio_path' in result:
            del result['temp_audio_path']
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Внутренняя ошибка сервера: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")
    finally:
        # Удаляем временные файлы в самом конце
        files_to_delete = [temp_audio_path]
        
        # Добавляем сконвертированный WAV файл если он есть
        if 'temp_wav_path' in locals() and temp_wav_path and temp_wav_path != temp_audio_path:
            files_to_delete.append(temp_wav_path)
        
        # Ищем другие временные файлы в папке temp
        try:
            import glob
            base_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
            # Ищем файлы созданные из этого файла
            converted_files = glob.glob(os.path.join("temp", f"converted_{base_name}*.wav"))
            processed_files = glob.glob(os.path.join("temp", f"processed_{base_name}*.wav"))
            files_to_delete.extend(converted_files)
            files_to_delete.extend(processed_files)
        except Exception as glob_error:
            logger.warning(f"Ошибка при поиске временных файлов: {glob_error}")
        
        # Удаляем все временные файлы
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    logger.info(f"Временный файл удален: {file_path}")
                except Exception as delete_error:
                    logger.warning(f"Не удалось удалить файл {file_path}: {delete_error}")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {
        "status": "healthy",
        "asr_model_loaded": asr_model is not None,
        "version": "1.0.0"
    }

if __name__ == '__main__':
    import uvicorn
    
    # Создаем папку temp при запуске
    os.makedirs("./temp", exist_ok=True)
    
    logger.info("Запуск HSKK Coach Server")
    logger.info("=" * 50)
    logger.info("Доступные эндпоинты:")
    logger.info("  GET  /        - Главная страница")
    logger.info("  POST /analyze - Анализ аудио")
    logger.info("  GET  /health  - Проверка здоровья")
    logger.info("  GET  /docs    - Swagger документация")
    logger.info("=" * 50)
    logger.info(f"Временные файлы будут сохраняться в: {os.path.abspath('./temp')}")
    logger.info("=" * 50)
    
    if asr_model:
        logger.info("ASR модель: загружена")
    else:
        logger.warning("ASR модель: не загружена (используется fallback)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)