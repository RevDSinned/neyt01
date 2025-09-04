import os, io, json, base64, pathlib, math, time, textwrap, random
from datetime import datetime
from typing import List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip
)
from openai import OpenAI

# Paths
ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
IMG_DIR = OUT / "img"
THUMB_PNG = OUT / "thumb.png"
VOICE_MP3 = OUT / "voice.mp3"
VIDEO_MP4 = OUT / "video.mp4"
SCRIPT_TXT = OUT / "script.txt"

# Clients / env
client = OpenAI()  # uses OPENAI_API_KEY from env

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")

YT_CLIENT_ID = os.environ.get("YT_CLIENT_ID", "")
YT_CLIENT_SECRET = os.environ.get("YT_CLIENT_SECRET", "")
YT_REFRESH_TOKEN = os.environ.get("YT_REFRESH_TOKEN", "")

# Utilities
def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def read_prompt(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")

def save_text(path: pathlib.Path, text: str):
    path.write_text(text, encoding="utf-8")

# LLM wrapper (GPT-5 via Responses; others via Chat Completions)
def chat(model: str, system_prompt: str, user_prompt: str) -> str:
    if model.startswith("gpt-5"):
        r = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (getattr(r, "output_text", "") or "").strip()
    else:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
        return (r.choices[0].message.content or "").strip()

# Topic → Script → Critique
def pick_topic(cfg: dict) -> str:
    sys = read_prompt("prompts/topic_generator_system.md")
    seeds = cfg.get("topic_seeds", [])
    ask = f"Channel: {cfg.get('channel_name','')}\nSeeds: {seeds}\nDuration: {cfg.get('duration_minutes',10)} min."
    return chat(cfg["writer_model"], sys, ask)

def write_script(cfg: dict, topic: str) -> str:
    sys = read_prompt("prompts/writer_system.md")
    avoid = ", ".join(cfg.get("avoid_terms", []))
    ask = (
        f"Topic: {topic}\n"
        f"Target duration: {cfg.get('duration_minutes',10)} minutes.\n"
        f"Avoid terms: {avoid}\n"
        f"Style: informative, friendly, natural voice-over.\n"
        f"Output: narration script only."
    )
    return chat(cfg["writer_model"], sys, ask)

def critique_script(cfg: dict, script: str) -> str:
    sys = read_prompt("prompts/critic_system.md")
    ask = (
        "Improve the narration for clarity and flow. Keep the same content length range. "
        "Return the revised script only.\n\nSCRIPT:\n" + script
    )
    return chat(cfg["critic_model"], sys, ask)

# Images
def generate_thumbnail(cfg: dict, topic: str, path: pathlib.Path):
    prompt = f"Minimal, high-contrast travel thumbnail about: {topic}. Bold composition, readable at small size."
    r = client.images.generate(model=cfg.get("image_model", "gpt-image-1"), prompt=prompt, size="1024x576")
    b64 = r.data[0].b64_json
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.save(path)

def wrap_text(draw, text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    return lines

def fallback_background(text: str, w=1920, h=1080) -> Image.Image:
    img = Image.new("RGB", (w, h), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    lines = wrap_text(draw, text, font, w - 200)
    y = h//3
    for ln in lines[:6]:
        draw.text((100, y), ln, fill=(240,240,240), font=font)
        y += 90
    return img

def download_pixabay_images(query: str, key: str, n: int = 8) -> List[pathlib.Path]:
    if not key:
        return []
    url = "https://pixabay.com/api/"
    params = dict(key=key, q=query, safesearch="true", image_type="photo",
                  orientation="horizontal", per_page=n)
    try:
        resp = requests.get(url, params=params, timeout=30)
        hits = resp.json().get("hits", [])
        paths = []
        for i, hit in enumerate(hits[:n], 1):
            img_url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not img_url: continue
            data = requests.get(img_url, timeout=30).content
            p = IMG_DIR / f"img_{i:02d}.jpg"
            p.write_bytes(data)
            paths.append(p)
        return paths
    except Exception:
        return []

# TTS (ElevenLabs)
def elevenlabs_tts(cfg: dict, text: str, out_path: pathlib.Path):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        # no TTS credentials; write empty file to avoid crash
        out_path.write_bytes(b"")
        return
    model_id = cfg.get("elevenlabs_model_id", "eleven_multilingual_v2")
    output_format = cfg.get("elevenlabs_output_format", "mp3_44100_128")
    voice_settings = cfg.get("elevenlabs_voice_settings", {})
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": f"audio/mpeg",
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

    # chunk long text for stability
    chunks: List[str] = []
    chunk = []
    total = 0
    for para in text.split("\n"):
        if not para.strip(): continue
        if total + len(para) > 2500:
            chunks.append("\n".join(chunk))
            chunk, total = [para], len(para)
        else:
            chunk.append(para); total += len(para)
    if chunk: chunks.append("\n".join(chunk))

    with open(out_path, "wb") as f:
        for part in chunks:
            payload = {
                "model_id": model_id,
                "text": part,
                "voice_settings": voice_settings,
                "output_format": output_format,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
            r.raise_for_status()
            for b in r.iter_content(chunk_size=16384):
                if b: f.write(b)
            # small gap between chunks
            time.sleep(0.2)

# Video assembly
def build_video(image_paths: List[pathlib.Path], audio_path: pathlib.Path, out_path: pathlib.Path):
    if not image_paths:
        # create one background slide
        bg = fallback_background("Nomad Economics")
        tmp = IMG_DIR / "bg.jpg"
        bg.save(tmp)
        image_paths = [tmp]

    # durations
    if audio_path.exists() and audio_path.stat().st_size > 0:
        audio = AudioFileClip(str(audio_path))
        total = max(10.0, audio.duration)
    else:
        audio = None
        total = 60.0

    per = total / len(image_paths)
    clips = []
    for p in image_paths:
        img_clip = ImageClip(str(p)).set_duration(per).resize(height=1080).on_color(size=(1920,1080), color=(0,0,0))
        clips.append(img_clip)
    video = concatenate_videoclips(clips, method="compose")
    if audio: video = video.set_audio(audio)
    video.write_videofile(str(out_path), fps=30, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    video.close()
    if audio: audio.close()

# YouTube upload (best-effort; do not fail build if missing)
def maybe_upload_to_youtube(cfg: dict, title: str, description: str, video_path: pathlib.Path, thumb_path: pathlib.Path):
    if not (YT_CLIENT_ID and YT_CLIENT_SECRET and YT_REFRESH_TOKEN):
        print("YouTube secrets not set; skipping upload.")
        return
    try:
        from youtube_uploader import upload_video
        upload_video(
            video_path=str(video_path),
            title=title,
            description=description,
            thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
            privacy_status=cfg.get("privacy_status", "public"),
            category_id=str(cfg.get("category_id", "19")),
            client_id=YT_CLIENT_ID,
            client_secret=YT_CLIENT_SECRET,
            refresh_token=YT_REFRESH_TOKEN,
        )
    except Exception as e:
        print(f"YouTube upload skipped: {e}")

def run():
    ensure_dirs()
    cfg = load_config()

    topic = pick_topic(cfg)
    script = write_script(cfg, topic)
    script = critique_script(cfg, script) or script
    save_text(SCRIPT_TXT, script)

    # Images
    imgs = download_pixabay_images(topic, PIXABAY_API_KEY, n=8)

    # TTS
    elevenlabs_tts(cfg, script, VOICE_MP3)

    # Video
    build_video(imgs, VOICE_MP3, VIDEO_MP4)

    # Thumbnail
    try:
        generate_thumbnail(cfg, topic, THUMB_PNG)
    except Exception as e:
        print(f"Thumbnail fallback: {e}")
        fallback_background(topic).save(THUMB_PNG)

    title = topic.strip()[:95]
    desc = f"{topic}\n\nGenerated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    maybe_upload_to_youtube(cfg, title, desc, VIDEO_MP4, THUMB_PNG)

if __name__ == "__main__":
    run()
