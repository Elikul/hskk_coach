from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
from funasr import AutoModel
import pypinyin
import re
import unicodedata

app = FastAPI()

# Загрузка ASR model (paraformer-zh-streaming с VAD для точной сегментации)
asr_model = AutoModel(
    model="paraformer-zh-streaming", 
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    disable_update=True
)


def normalize_text(text):
    # Убрать пробелы, пунктуацию и нормализовать
    text = re.sub(r"[^\w]", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text


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


def classify_tones_from_segments(f0, sr, segments, reference_text):
    # Классификация тонов с использованием точных временных меток из ASR сегментов
    tones = []
    
    # Получаем временные метки для каждого кадра F0
    frame_duration = 0.01  # 10ms per frame (стандартно для F0)
    total_duration = len(f0) * frame_duration
    
    # Фильтруем нулевые значения
    f0_valid = f0[f0 > 0]
    if len(f0_valid) == 0:
        return [0] * len(reference_text)
    
    global_mean = np.mean(f0_valid)
    global_std = np.std(f0_valid)
    
    # Обрабатываем каждый сегмент из ASR
    for segment in segments:
        if "text" not in segment:
            continue
            
        segment_text = normalize_text(segment["text"])
        segment_start = segment.get("start", 0)
        segment_end = segment.get("end", segment_start + 1.0)
        
        # Конвертируем время в индексы F0
        start_frame = int(segment_start / frame_duration)
        end_frame = int(segment_end / frame_duration)
        
        # Ограничиваем границы
        start_frame = max(0, min(start_frame, len(f0)))
        end_frame = max(0, min(end_frame, len(f0)))
        
        if start_frame >= end_frame:
            continue
        
        # Получаем F0 для этого сегмента
        pitch_segment = f0[start_frame:end_frame]
        
        if len(pitch_segment) == 0:
            tones.extend([0] * len(segment_text))
            continue

        # Фильтруем нулевые значения
        valid_pitch = pitch_segment[pitch_segment > 0]
        if len(valid_pitch) < 2:
            tones.extend([0] * len(segment_text))
            continue

        # Анализируем тон для всего сегмента
        segment_tone = analyze_pitch_contour(valid_pitch, global_mean, global_std)
        
        # Назначаем тот же тон всем символам в сегменте
        tones.extend([segment_tone] * len(segment_text))
    
    # Если количество тонов не совпадает с эталоном, дополняем или обрезаем
    while len(tones) < len(reference_text):
        tones.append(0)
    
    return tones[:len(reference_text)]


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
    # Простая и надежная классификация тонов на основе F0 контура (запасной вариант)
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


def plot_f0(audio_path, output_dir, reference_text=None, ref_tones=None):
    y, sr = librosa.load(audio_path, sr=16000, dtype=np.float64)
    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    
    plt.figure(figsize=(12, 6))
    
    # График F0 пользователя
    plt.subplot(2, 1, 1)
    plt.plot(t, f0, 'b-', label='Пользователь F0', alpha=0.7)
    
    # Добавляем сегменты тонов пользователя
    if reference_text and ref_tones:
        num_syllables = len(reference_text)
        segment_len = len(t) // num_syllables if num_syllables > 0 else 1
        
        colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'blue', 4: 'orange'}
        tone_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
        
        for i, (char, tone) in enumerate(zip(reference_text, ref_tones)):
            start_time = i * segment_len / len(t) * t[-1]
            end_time = (i + 1) * segment_len / len(t) * t[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors.get(tone, 'gray'))
            plt.text((start_time + end_time) / 2, max(f0) * 0.9, f'{char}\n{tone_names.get(tone, tone)}', 
                    ha='center', va='center', fontsize=8)
    
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title("F0 пользователя с сегментацией тонов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График эталонных тонов (синтетический)
    plt.subplot(2, 1, 2)
    if reference_text and ref_tones:
        # Создаем синтетический F0 контур для эталонных тонов
        ref_f0 = generate_reference_f0(reference_text, ref_tones, len(t), t)
        plt.plot(t, ref_f0, 'r-', label='Эталон F0', linewidth=2, alpha=0.8)
        
        # Добавляем сегменты эталонных тонов
        num_syllables = len(reference_text)
        segment_len = len(t) // num_syllables if num_syllables > 0 else 1
        
        colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'blue', 4: 'orange'}
        tone_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
        
        for i, (char, tone) in enumerate(zip(reference_text, ref_tones)):
            start_time = i * segment_len / len(t) * t[-1]
            end_time = (i + 1) * segment_len / len(t) * t[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors.get(tone, 'gray'))
            plt.text((start_time + end_time) / 2, max(ref_f0) * 0.9, f'{char}\n{tone_names.get(tone, tone)}', 
                    ha='center', va='center', fontsize=8)
    
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title("Эталонный F0 контур")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".wav", "_f0_comparison.png"))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return f0, sr, output_path


def generate_reference_f0(text, tones, length, time_array):
    """Генерирует синтетический F0 контур для эталонных тонов"""
    f0_ref = np.zeros(length)
    num_syllables = len(text)
    segment_len = length // num_syllables if num_syllables > 0 else 1
    
    # Базовые частоты для тонов (в Гц)
    base_freqs = {0: 150, 1: 250, 2: 180, 3: 120, 4: 220}
    
    for i, (char, tone) in enumerate(zip(text, tones)):
        start = i * segment_len
        end = (i + 1) * segment_len
        if end > length:
            end = length
        
        segment_length = end - start
        base_freq = base_freqs.get(tone, 150)
        
        # Генерируем контур в зависимости от тона
        if tone == 0:  # нейтральный
            f0_ref[start:end] = base_freq
        elif tone == 1:  # высокий плоский
            f0_ref[start:end] = base_freq
        elif tone == 2:  # поднимающийся
            t_segment = np.linspace(0, 1, segment_length)
            f0_ref[start:end] = base_freq * (1 + 0.5 * t_segment)
        elif tone == 3:  # низкий с подъемом
            t_segment = np.linspace(0, 1, segment_length)
            f0_ref[start:end] = base_freq * (1 - 0.3 * np.sin(np.pi * t_segment))
        elif tone == 4:  # падающий
            t_segment = np.linspace(0, 1, segment_length)
            f0_ref[start:end] = base_freq * (1.5 - 0.5 * t_segment)
    
    # Добавляем небольшой шум для реалистичности
    noise = np.random.normal(0, 5, length)
    f0_ref += noise
    f0_ref = np.maximum(f0_ref, 0)  # Убираем отрицательные значения
    
    return f0_ref


@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...), reference_text: str = Form(...), output_dir: str = Form("./temp")):
    # Сохранить видео из временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        audio_path = temp_audio.name

    try:
        # Взять текст из аудио, с помощью paraformer-zh-streaming с VAD
        chunk_size = [0, 10, 5]  # 600ms chunks
        encoder_chunk_look_back = 4
        decoder_chunk_look_back = 1
        
        # Загружаем аудио для потоковой обработки
        y, sr = librosa.load(audio_path, sr=16000)
        chunk_stride = chunk_size[1] * 960  # 600ms в сэмплах
        
        cache = {}
        total_chunks = int(len(y) / chunk_stride + 1)
        all_segments = []
        
        for i in range(total_chunks):
            speech_chunk = y[i * chunk_stride:(i + 1) * chunk_stride]
            is_final = i == total_chunks - 1
            
            try:
                res = asr_model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back
                )
                
                if res and len(res) > 0:
                    all_segments.extend(res)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                # Если потоковая обработка не работает, используем обычную
                break
        
        # Если потоковая обработка не сработала, используем обычную ASR
        if not all_segments:
            fallback_model = AutoModel(model="paraformer-zh", disable_update=True)
            text_result = fallback_model.generate(input=audio_path)
            user_text = normalize_text(text_result[0]["text"])
        else:
            # Объединяем все сегменты
            user_text = normalize_text("".join([seg.get("text", "") for seg in all_segments]))
            
        reference_text = normalize_text(reference_text)
        
        reference_text = normalize_text(reference_text)

        # Получить пиньинь из текста
        ref_pinyin_tone3_list = pypinyin.pinyin(reference_text, style=pypinyin.TONE3)
        ref_pinyin = " ".join(["".join(word) for word in ref_pinyin_tone3_list])
        user_pinyin_tone3_list = pypinyin.pinyin(user_text, style=pypinyin.TONE3)

        ref_tones = extract_tones_from_pinyin(ref_pinyin)

        # Извлечь f0 and классифицировать пользовательские тона
        f0, sr, plot_path = plot_f0(audio_path, output_dir, reference_text, ref_tones)
        
        # Используем сегменты из ASR для точной сегментации тонов
        if all_segments:
            user_tones = classify_tones_from_segments(f0, sr, all_segments, reference_text)
        else:
            user_tones = classify_tones_from_f0(f0, sr, reference_text)

        # Сравнить тона
        comparison = []
        for i, (u, r) in enumerate(zip(user_tones, ref_tones)):
            char_user = user_text[i]
            char_ref = reference_text[i]
            # Для user: диакритика + цифра из аудио
            if i < len(user_pinyin_tone3_list):
                pinyin_user_base = "".join(user_pinyin_tone3_list[i])
                pinyin_user_with_tone = pinyin_user_base[:-1] + str(u) if pinyin_user_base[-1].isdigit() else pinyin_user_base + str(u)
            else:
                pinyin_user_with_tone = f"{char_user}({u})"
            # Для ref: диакритика + цифра из текста
            pinyin_ref_base = "".join(ref_pinyin_tone3_list[i])
            pinyin_ref = pinyin_ref_base[:-1] + str(r) if pinyin_ref_base[-1].isdigit() else pinyin_ref_base + str(r)
            if u == r:
                comparison.append(f"Верно: {char_ref} ({pinyin_ref})")
            else:
                comparison.append(
                    f"Неверно: сказано {char_user} ({pinyin_user_with_tone}), должно быть {char_ref} ({pinyin_ref})"
                )

        return {
            "reference_text": reference_text,
            "user_text": user_text,
            "reference_tones": ref_tones,
            "user_tones": user_tones,
            "comparison": comparison,
            "f0_plot": plot_path,
        }

    finally:
        os.unlink(audio_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
