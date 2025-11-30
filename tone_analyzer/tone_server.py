import tempfile
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from pypinyin import pinyin, Style
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import torch

app = FastAPI(title="Chinese Tone Analyzer", version="1.0")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("models/tone_cnn.pth")
device = torch.device("cpu") 

class ToneCNN(torch.nn.Module):
    def __init__(self, input_len=50, num_classes=5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * (input_len // 4), 128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Модель не найдена: {MODEL_PATH.absolute()}. "
        "Поместите обученную модель 'tone_cnn.pth' в папку 'models'"
    )

model = ToneCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Модель загружена: {MODEL_PATH}")


def get_reference_tones(text: str) -> List[int]:
    """Извлекает только номера тонов из китайского текста"""
    py_list = pinyin(text, style=Style.TONE3, heteronym=False)
    tones = []
    for item in py_list:
        syl = item[0]
        tone = 5
        if syl[-1].isdigit():
            tone = int(syl[-1])
        tones.append(tone)
    return tones


def extract_f0(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Извлекает F0 с помощью librosa.pyin"""
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    segment = audio[start_idx:end_idx]
    if len(segment) == 0:
        return np.array([])
    f0, _, _ = librosa.pyin(
        segment,
        fmin=librosa.note_to_hz("C2"),  # ~65 Гц
        fmax=librosa.note_to_hz("C7"),  # ~2093 Гц
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )
    return f0[~np.isnan(f0)]


def predict_tone(f0_contour: np.ndarray) -> int:
    """Предсказывает тон с помощью нейросети"""
    if len(f0_contour) < 2:
        return 5
    log_f0 = np.log(f0_contour[f0_contour > 0])
    if len(log_f0) < 2:
        return 5
    x_old = np.linspace(0, 1, len(log_f0))
    x_new = np.linspace(0, 1, 50)
    f0_interp = np.interp(x_new, x_old, log_f0)
    f0_norm = (f0_interp - f0_interp.mean()) / (f0_interp.std() + 1e-8)
    with torch.no_grad():
        x_tensor = torch.tensor(f0_norm, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x_tensor)
        pred = logits.argmax(dim=1).item()
        return pred + 1  # 0–4 → 1–5


def plot_analysis(
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    user_tones: List[int],
    ref_tones: List[int],
    output_path: Path,
):
    """Генерирует график F0 с разметкой тонов"""
    f0_full, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )
    times = librosa.times_like(f0_full, sr=sr, hop_length=256)
    plt.figure(figsize=(12, 5))
    plt.plot(times, f0_full, color="lightgray", label="F0 (librosa.pyin)")
    colors = ["green" if u == r else "red" for u, r in zip(user_tones, ref_tones)]
    for i, ((start, end), color) in enumerate(zip(segments, colors)):
        plt.axvspan(start, end, color=color, alpha=0.3)
        mid = (start + end) / 2
        y_pos = np.nanmax(f0_full) * 0.95 - i * 20
        plt.text(mid, y_pos, f"{user_tones[i]} vs {ref_tones[i]}", ha="center", fontsize=9, color=color)
    plt.xlabel("Время (с)")
    plt.ylabel("F0 (Гц)")
    plt.title("Анализ тонов: зелёный — верно, красный — ошибка")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@app.post("/analyze_tones")
async def analyze_tones(audio: UploadFile = File(...), reference_text: str = Form(...)):
    try:
        # Валидация входных данных
        if not reference_text.strip():
            return JSONResponse(status_code=400, content={"error": "Текст не может быть пустым"})

        # Создание рабочей директории
        work_dir = TEMP_DIR / next(tempfile._get_candidate_names())
        work_dir.mkdir()

        # Сохранение и загрузка аудио
        audio_path = work_dir / "audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        sf.write(audio_path, y, sr, format="WAV", subtype="PCM_16")

        # Получение эталонных тонов
        ref_tones = get_reference_tones(reference_text)
        if not ref_tones:
            return JSONResponse(status_code=400, content={"error": "Не удалось извлечь тоны из текста"})

        # Сегментация и предсказание
        duration = len(y) / sr
        n_syllables = len(ref_tones)
        segments = []
        user_tones = []

        for i in range(n_syllables):
            start = i * duration / n_syllables
            end = (i + 1) * duration / n_syllables
            segments.append((start, end))
            f0_vals = extract_f0(y, sr, start, end)
            tone = predict_tone(f0_vals)
            user_tones.append(tone)

        # Обратная связь
        feedback = [
            f"Слог {i+1}: сказано {u_t}-й тон → {'Верно' if u_t == r_t else f'Нужно: {r_t}-й тон'}"
            for i, (u_t, r_t) in enumerate(zip(user_tones, ref_tones))
        ]

        # Визуализация
        image_path = work_dir / "analysis.png"
        plot_analysis(y, sr, segments, user_tones, ref_tones, image_path)

        return JSONResponse({
            "reference_text": reference_text,
            "reference_tones": ref_tones,
            "user_tones": user_tones,
            "feedback": feedback,
            "image_url": f"/results/{work_dir.name}/analysis.png"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/results/{job_id}/{filename}")
async def get_result_file(job_id: str, filename: str):
    file_path = TEMP_DIR / job_id / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Файл не найден"})
    return FileResponse(file_path)