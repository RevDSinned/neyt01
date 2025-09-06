import os, json, math, time, base64, io, pathlib, random, re, shutil
from datetime import datetime
from typing import List

import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from openai import OpenAI

# ---------- Paths ----------
ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
IMG_DIR = OUT / "img"
THUMB = OUT / "thumb.png"
VOICE_MP3 = OUT / "voice.mp3"
VIDEO_MP4 = OUT / "video.mp4"
SCRIPT_TXT = OUT / "script.txt"
KEYWORDS_TXT = OUT / "keywords.txt"

# ---------- Env / Clients ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")

YT_CLIENT_ID = os.environ.get("YT_CLIENT_ID", "")
YT_CLIENT_SECRET = os.environ.get("YT_CLIENT_SECRET", "")
YT_REFRESH_TOKEN = os.environ.get("YT_REFRESH_TOKEN", "")

# ---------- Utils ----------
def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def read_prompt(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")

def save_text(path: pathlib.Path, text: str):
    path.write_text(text, encoding="utf-8")

def seconds_from_mp3(path: pathlib.Path) -> float:
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    with AudioFileClip(str(path)) as a:
        return float(a.duration)

# ---------- OpenAI (GPT-5 via Responses) ----------
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
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

# ---------- Script workflow ----------
def system_writer(): return read_prompt("prompts/writer_system.md")
def system_critic(): return read_prompt("prompts/critic_system.md")
def system_topic():  return read_prompt("prompts/topic_generator_system.md")

def pick_topic(cfg) -> str:
    seeds = cfg.get("topic_seeds", [])
    avoid = ", ".join(cfg.get("avoid_terms", []))
    sys = system_topic()
    ask = f"Channel: {cfg.get('channel_name','')}\nSeeds: {seeds}\nAvoid: {avoid}\nReturn one specific topic (no quotes)."
    return chat(cfg["writer_model"], sys, ask).splitlines()[0].strip()

def write_script(cfg, topic: str) -> str:
    sys = system_writer()
    ask = (
        f"Write a natural, human-sounding narration for a {cfg.get('duration_minutes',1)}-minute YouTube video titled:\n"
        f"{topic}\n\n"
        "Hook in 10s, clear structure, concrete prices, crisp sentences. Output narration only."
    )
    return chat(cfg["writer_model"], sys, ask)

def critique_and_revise(cfg, script: str) -> str:
    sys = system_critic()
    ask = "Improve clarity, pacing, and retention cues. Keep length. Return revised narration only.\n\n" + script
    return chat(cfg["critic_model"], sys, ask)

def extract_broll_keywords(cfg, script: str) -> List[str]:
    ask = "List 20 concise, comma-separated b-roll search keywords (no numbering)."
    resp = chat(cfg["critic_model"], "You generate concise keyword lists.", ask + "\n\n" + script)
    parts = [p.strip() for p in re.split(r"[,\n]", resp) if p.strip()]
    return [p[:40] for p in parts][:20]

def title_desc_tags(cfg, topic: str, script: str):
    ask = """Create:
1) Title (<55 chars)
2) Two-paragraph description with value + CTA
3) 12 short SEO tags (comma-separated)

Return exactly:
TITLE:
...
DESCRIPTION:
...
TAGS:
..."""
    resp = chat(cfg["writer_model"], "You are a YouTube strategist.", ask + "\n\n" + script)
    title_m = re.search(r"TITLE:\s*(.+)", resp)
    desc_m  = re.search(r"DESCRIPTION:\s*(.*?)(?:\nTAGS:|$)", resp, flags=re.S|re.I)
    tags_m  = re.search(r"TAGS:\s*(.+)", resp)
    title = (title_m.group(1).strip() if title_m else topic)[:55]
    description = (desc_m.group(1).strip() if desc_m else topic)
    tags = [t.strip() for t in re.split(r"[,\n]", tags_m.group(1))] if tags_m else []
    return title, description, tags[:15]

# ---------- Images ----------
def pexels_search(query, per_page=40):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY, "User-Agent": "nomad-econ-yt"}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    r = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=30)
    if r.status_code != 200: return []
    data = r.json()
    return [p["src"]["large"] for p in data.get("photos", []) if "src" in p and "large" in p["src"]]

def pixabay_search(query, per_page=40):
    if not PIXABAY_API_KEY: return []
    params = {
        "key": PIXABAY_API_KEY, "q": query, "image_type": "photo",
        "per_page": per_page, "safesearch": "true", "orientation": "horizontal"
    }
    r = requests.get("https://pixabay.com/api/", params=params, timeout=30)
    if r.status_code != 200: return []
    data = r.json()
    return [h.get("largeImageURL") or h.get("webformatURL") for h in data.get("hits", [])]

def download_images(urls, limit=120) -> List[pathlib.Path]:
    saved = []
    for i, u in enumerate(urls[:limit]):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 200:
                f = IMG_DIR / f"img_{i:03d}.jpg"
                f.write_bytes(r.content)
                saved.append(f)
        except Exception:
            pass
    return saved

def fetch_broll(keywords: List[str], need_images: int) -> List[pathlib.Path]:
    urls: List[str] = []
    for kw in keywords:
        urls += pexels_search(kw, per_page=10)
        urls += pixabay_search(kw, per_page=10)
        if len(urls) >= need_images * 2:
            break
    random.shuffle(urls)
    return download_images(urls, limit=need_images)

# ---------- ElevenLabs TTS ----------
def elevenlabs_tts(cfg: dict, text: str, out_path: pathlib.Path):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        out_path.write_bytes(b"")
        return

    model_id = cfg.get("elevenlabs_model_id", "eleven_multilingual_v2")
    output_format = cfg.get("elevenlabs_output_format", "mp3_44100_128")
    voice_settings = cfg.get("elevenlabs_voice_settings", {
        "stability": 0.40, "similarity_boost": 0.75, "style": 0.25, "use_speaker_boost": True
    })

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "User-Agent": "nomad-econ-yt"
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

    def split_text(s: str, maxlen=2400):
        s = re.sub(r"\s+", " ", s).strip()
        while len(s) > maxlen:
            cut = s.rfind(".", 0, maxlen)
            if cut < 0: cut = maxlen
            yield s[:cut+1].strip()
            s = s[cut+1:].strip()
        if s: yield s

    # Write/append streaming chunks
    out_path.write_bytes(b"")
    for part in split_text(text):
        payload = {
            "text": part,
            "model_id": model_id,
            "voice_settings": voice_settings,
            "output_format": output_format
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
        if r.status_code != 200:
            # Log server message then continue gracefully
            msg = r.text[:500]
            print(f"[ElevenLabs 400] {msg}")
            raise RuntimeError(f"ElevenLabs error {r.status_code}")
        with open(out_path, "ab") as f:
            for chunk in r.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
        time.sleep(0.2)

# ---------- Video rendering ----------
def fallback_background(text: str, w=1920, h=1080) -> Image.Image:
    img = Image.new("RGB", (w, h), (20, 20, 24))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except:
        font = ImageFont.load_default()
    d.text((80, h//3), text[:60], fill=(235,235,235), font=font)
    return img

def build_video(image_paths: List[pathlib.Path], audio_path: pathlib.Path, out_path: pathlib.Path):
    if not image_paths:
        tmp = IMG_DIR / "bg.jpg"
        fallback_background("Nomad Economics").save(tmp)
        image_paths = [tmp]

    if audio_path.exists() and audio_path.stat().st_size > 0:
        audio = AudioFileClip(str(audio_path))
        total = max(1.0, audio.duration)
    else:
        audio = None
        total = 60.0

    per = total / max(1, len(image_paths))
    clips = [ImageClip(str(p)).set_duration(per).resize(height=1080).on_color(size=(1920,1080), color=(0,0,0))
             for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    if audio: video = video.set_audio(audio)
    video.write_videofile(str(out_path), fps=30, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    video.close()
    if audio: audio.close()
    for c in clips:
        try: c.close()
        except: pass
    return out_path

# ---------- Thumbnail ----------
def make_thumbnail(title: str, cfg: dict) -> pathlib.Path:
    try:
        prompt = f"Bold, high-contrast photo real travel thumbnail background, no text. Topic: {title}"
        r = client.images.generate(model=cfg.get("image_model", "gpt-image-1"), prompt=prompt, size="1536x1024")
        b64 = r.data[0].b64_json
        Image.open(io.BytesIO(base64.b64decode(b64))).save(THUMB, "PNG")
    except Exception as e:
        print(f"Thumbnail gen failed: {e}")
        fallback_background(title, 1280, 720).save(THUMB, "PNG")
    return THUMB

from pathlib import Path

# ---------- Optional YouTube upload ----------
def maybe_upload_to_youtube(cfg: dict, title: str, description: str, video_path: Path, thumb_path: Path | None):
    if not all([YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN]):
        print("YouTube secrets not set; skipping upload.")
        return None

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}; skipping upload.")
        return None
    if thumb_path and not Path(thumb_path).exists():
        print(f"Thumbnail not found: {thumb_path}; will upload without a thumbnail.")
        thumb_path = None

    privacy = str(cfg.get("privacy_status", "public")).lower()
    if privacy not in {"public", "unlisted", "private"}:
        privacy = "public"
    category_id = str(cfg.get("category_id", "19"))

    raw_tags = cfg.get("tags", [])
    if isinstance(raw_tags, str):
        tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, (list, tuple, set)):
        tags_list = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tags_list = []
    # trim to YouTube's ~500-char total across tags (commas count)
    total = 0; trimmed = []
    for t in tags_list:
        sep = 1 if trimmed else 0
        if total + sep + len(t) > 500:
            break
        trimmed.append(t); total += sep + len(t)
    tags_list = trimmed

    try:
        from youtube_uploader import upload_video
    except Exception as e:
        print(f"YouTube uploader import failed: {e}; skipping upload.")
        return None

    params = set(inspect.signature(upload_video).parameters)
    # pass only common content fields; DO NOT pass client_id/client_secret/refresh_token
    candidate = {
        "title": title[:100],
        "description": description[:5000],
        "thumbnail_path": str(thumb_path) if thumb_path else None,
        "thumbnail": str(thumb_path) if thumb_path else None,
        "thumb": str(thumb_path) if thumb_path else None,
        "privacy_status": privacy,
        "privacy": privacy,
        "category_id": category_id,
        "category": category_id,
        "categoryId": category_id,
        "tags": tags_list,
    }
    kwargs = {k: v for k, v in candidate.items() if k in params and v is not None}

    try:
        video_id = upload_video(str(video_path), **kwargs)
        print(f"YouTube upload complete: https://youtu.be/{video_id}")
        return video_id
    except Exception as e:
        print(f"YouTube upload skipped: {e}")
        return None
        
# ---------- Orchestration ----------
def run():
    ensure_dirs()
    cfg = load_config()

    print("→ Picking topic…")
    topic = pick_topic(cfg)
    print("TOPIC:", topic)

    print("→ Writing script…")
    script = write_script(cfg, topic)

    print("→ Critique & revise…")
    script2 = critique_and_revise(cfg, script)
    save_text(SCRIPT_TXT, script2)

    print("→ Extracting b-roll keywords…")
    kws = extract_broll_keywords(cfg, script2)
    save_text(KEYWORDS_TXT, ", ".join(kws))

    print("→ Narration (ElevenLabs)…")
    elevenlabs_tts(cfg, script2, VOICE_MP3)
    dur = max(1.0, seconds_from_mp3(VOICE_MP3))
    print(f"Narration length: {dur:.1f}s")

    print("→ Fetching images…")
    imgs = fetch_broll(kws, need_images=math.ceil(dur/6))
    print("Images downloaded:", len(imgs))

    print("→ Rendering video…")
    build_video(imgs, VOICE_MP3, VIDEO_MP4)

    print("→ Title, description, tags…")
    title, description, tags = title_desc_tags(cfg, topic, script2)

    print("→ Thumbnail…")
    make_thumbnail(title, cfg)

    print("→ Upload (if secrets present)…")
    maybe_upload_to_youtube(cfg, title, description, VIDEO_MP4, THUMB)

if __name__ == "__main__":
    run()
