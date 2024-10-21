import time
import gradio as gr
from torch import device as torch_device, cuda
from transformers import pipeline

# 檢測設備類型
device = torch_device("cuda" if cuda.is_available() else "cpu")

pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    chunk_length_s=30,
    device=device,
)


# 定義calc_time用來計算函數花費時間
def calc_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    return result

# 定義transcribe用來收集聲音
def transcribe(audio):
    voice = calc_time(pipe, audio, batch_size=8, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    # text = pipe(audio, batch_size=8, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    return voice

# 快速處理介面用
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text"
)

if __name__ == "__main__":
    print(f"Using device: {device}")
    demo.launch()
