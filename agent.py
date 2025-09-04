import os, json, math, time, base64, io, pathlib, random, textwrap, re, shutil, tempfile
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from openai import OpenAI

# ---------- Paths / Config ----------
ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
IMG_DIR = OUT / "img"
THUMB = OUT / "thumb.png"
VOICE_MP3 = OUT / "voice.mp3"
VIDEO_MP4 = OUT / "video.mp4"

def load_config():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return cfg

# ---------- Secrets (env) ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# YouTube uploader relies on:
# YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN (env)

# ElevenLabs
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

# Stock providers (optional)
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY")

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Helpers ----------
def ensure_dirs():
    for p in [OUT, IMG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def save_text(path: pathlib.Path, text: str):
    path.write_text(text, encoding="utf-8")

def read_md(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", s).strip()
    return re.sub(r"\s+", " ", s)

def seconds_from_mp3(path: pathlib.Path) -> float:
    # moviepy reads duration from file metadata
    with AudioFileClip(str(path)) as a:
        return float(a.duration)

# ---------- OpenAI calls ----------
def chat(model: str, system: str, user: str) -> str:
    """
    Uses Chat Completions; falls back to Responses if needed.
    """
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        r = client.responses.create(
            model=model,
            input=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        # best-effort extraction
        if hasattr(r, "output") and len(r.output) and hasattr(r.output[0], "content"):
            return "".join([c.text for c in r.output[0].content if getattr(c, "type", "")=="output_text"]).strip()
        return str(r)

def gen_image(prompt: str, path: pathlib.Path, size="1024x1024"):
    r = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
    b64 = r.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    path.write_bytes(img_bytes)

# ---------- ElevenLabs TTS ----------
def tts_elevenlabs(text: str, out_path: pathlib.Path, cfg: dict):
    assert ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID, "Missing ELEVENLABS secrets"
    model_id = cfg.get("elevenlabs_model_id", "eleven_multilingual_v2")
    output_format = cfg.get("elevenlabs_output_format", "mp3_44100_128")
    voice_settings = cfg.get("elevenlabs_voice_settings", {
        "stability": 0.40, "similarity_boost": 0.75, "style": 0.25, "use_speaker_boost": True
    })

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "user-agent": "nomad-econ-yt"
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    # chunk long text to stay within request limits
    def chunks(s, maxlen=1800):
        s = re.sub(r"\s+", " ", s).strip()
        while len(s) > maxlen:
            cut = s.rfind(".", 0, maxlen)
            if cut < 0: cut = maxlen
            yield s[:cut+1].strip()
            s = s[cut+1:].strip()
        if s: yield s

    seg_files = []
    for i, part in enumerate(chunks(text)):
        payload = {
            "text": part,
            "model_id": model_id,
            "voice_settings": voice_settings,
            "output_format": output_format
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        seg = OUT / f"voice_seg_{i:02d}.mp3"
        seg.write_bytes(resp.content)
        seg_files.append(seg)

    if len(seg_files) == 1:
        shutil.copy(seg_files[0], out_path)
        return

    clips = [AudioFileClip(str(p)) for p in seg_files]
    final = concatenate_videoclips([])  # placeholder to unify handles
    audio = clips[0]
    for c in clips[1:]:
        audio = audio.append(c)
    audio.write_audiofile(str(out_path), codec="mp3")
    for c in clips:
        c.close()

# ---------- Image search / download ----------
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
    return [h["largeImageURL"] for h in data.get("hits", []) if "largeImageURL" in h]

def download_images(urls, limit=120):
    saved = []
    for i, u in enumerate(urls[:limit]):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 200:
                f = IMG_DIR / f"img_{i:03d}.jpg"
                f.write_bytes(r.content)
                saved.append(f)
        except Exception:
            continue
    return saved

# ---------- Rendering ----------
def assemble_video(narration_mp3: pathlib.Path, images: list[pathlib.Path]) -> pathlib.Path:
    if not images:
        # fallback: solid background frame
        bg = Image.new("RGB", (1280, 720), (18, 18, 24))
        fp = IMG_DIR / "fallback.jpg"
        bg.save(fp, "JPEG")
        images = [fp]

    dur = max(1.0, seconds_from_mp3(narration_mp3))
    per = dur / len(images)

    clips = [ImageClip(str(p)).set_duration(per).resize((1280,720)) for p in images]
    video = concatenate_videoclips(clips, method="chain")
    video = video.set_audio(AudioFileClip(str(narration_mp3)))
    video.write_videofile(str(VIDEO_MP4), fps=30, codec="libx264", audio_codec="aac", bitrate="4000k")
    video.close()
    for c in clips:
        try: c.close()
        except: pass
    return VIDEO_MP4

# ---------- Thumbnail ----------
def make_thumbnail(title: str, cfg: dict) -> pathlib.Path:
    try:
        prompt = f"Bold high-contrast travel thumbnail background, abstract world map, neon accents, cinematic lighting, no text, 16:9 composition."
        gen_image(prompt, THUMB, size="1024x1024")
        # overlay title
        img = Image.open(THUMB).convert("RGB").resize((1280, 720))
    except Exception:
        img = Image.new("RGB", (1280, 720), (25,25,30))

    draw = ImageDraw.Draw(img)
    try:
        # Use a generic sans if no font installed in runner
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except:
        font = ImageFont.load_default()

    text = title[:40].upper()
    # shadow + text
    for off in [(4,4),(2,2),(1,1)]:
        draw.text((70+off[0], 520+off[1]), text, fill=(0,0,0), font=font)
    draw.text((70, 520), text, fill=(245,245,245), font=font)
    img.save(THUMB, "PNG")
    return THUMB

# ---------- Prompts ----------
def system_writer():
    return (ROOT / "prompts" / "writer_system.md").read_text(encoding="utf-8")

def system_critic():
    return (ROOT / "prompts" / "critic_system.md").read_text(encoding="utf-8")

def system_topic():
    return (ROOT / "prompts" / "topic_generator_system.md").read_text(encoding="utf-8")

# ---------- Core pipeline ----------
def pick_topic(cfg) -> str:
    seeds = cfg.get("topic_seeds", [])
    avoid = ", ".join(cfg.get("avoid_terms", []))
    sys = system_topic()
    ask = f"""Channel pillars: travel + money. Duration target: {cfg.get('duration_minutes',10)} minutes.
Seed ideas: {seeds}.
Avoid: {avoid}.
Return a single irresistible, specific topic only (no quotes)."""
    topic = chat(cfg["writer_model"], sys, ask)
    return topic.strip().splitlines()[0]

def write_script(cfg, topic: str) -> str:
    sys = system_writer()
    ask = f"""Write a tight, human-sounding 10-minute script for YouTube titled:
{topic}

Rules:
- Hook in first 10 seconds.
- Clear structure with timestamped chapter markers.
- Crisp sentences (12–20 words), zero fluff.
- Explain costs with real numbers and ranges.
- End with a concise summary + call to action."""
    script = chat(cfg["writer_model"], sys, ask)
    return script

def critique_and_revise(cfg, script: str) -> str:
    sys = system_critic()
    ask = f"Critique and improve this script for retention and clarity. Keep structure, fix pacing and add any missing cost details.\n\n{script}"
    improved = chat(cfg["critic_model"], sys, ask)
    return improved

def extract_broll_keywords(cfg, script: str) -> list[str]:
    ask = "List 20 short, comma-separated b-roll search keywords (no numbering) for the script."
    resp = chat(cfg["critic_model"], "You generate concise keyword lists.", ask + "\n\n" + script)
    parts = [p.strip() for p in re.split(r"[,\n]", resp) if p.strip()]
    # keep short phrases only
    return [p[:40] for p in parts][:20]

def title_desc_tags(cfg, topic: str, script: str):
    ask = """Create:
1) A compelling YouTube title under 55 chars.
2) A two-paragraph description with value + concise CTA.
3) 12 short SEO tags (comma-separated).

Return exactly:
TITLE:
...
DESCRIPTION:
...
TAGS:
..."""
    resp = chat(cfg["writer_model"], "You are a YouTube strategist.", ask + "\n\n" + script)
    title = re.search(r"TITLE:\s*(.+)", resp)
    desc = re.search(r"DESCRIPTION:\s*(.*?)(?:TAGS:|$)", resp, flags=re.S|re.I)
    tags = re.search(r"TAGS:\s*(.+)", resp)
    title = title.group(1).strip() if title else topic[:55]
    description = desc.group(1).strip() if desc else topic
    tag_list = [t.strip() for t in re.split(r"[,\n]", tags.group(1))] if tags else []
    return title, description, tag_list[:15]

def fetch_broll(keywords: list[str], need_images: int = 120) -> list[pathlib.Path]:
    urls = []
    # alternate providers to spread rate limits
    for i, kw in enumerate(keywords):
        urls += pexels_search(kw, per_page=10)
        urls += pixabay_search(kw, per_page=10)
        if len(urls) >= need_images * 2:
            break
    random.shuffle(urls)
    return download_images(urls, limit=need_images)

def run():
    ensure_dirs()
    cfg = load_config()

    print("→ Picking topic…")
    topic = pick_topic(cfg)
    print("TOPIC:", topic)

    print("→ Writing script…")
    script = write_script(cfg, topic)
    print("Script chars:", len(script))

    print("→ Critique & revise…")
    script2 = critique_and_revise(cfg, script)
    save_text(OUT / "script.txt", script2)

    print("→ Extracting b-roll keywords…")
    kws = extract_broll_keywords(cfg, script2)
    save_text(OUT / "keywords.txt", ", ".join(kws))

    print("→ Narration (ElevenLabs)…")
    tts_elevenlabs(script2, VOICE_MP3, cfg)
    dur = seconds_from_mp3(VOICE_MP3)
    print(f"Narration length: {dur:.1f}s")

    print("→ Fetching images…")
    imgs = fetch_broll(kws, need_images=math.ceil(dur/6))
    print("Images downloaded:", len(imgs))

    print("→ Rendering video…")
    assemble_video(VOICE_MP3, imgs)

    print("→ Title, description, tags…")
    title, description, tags = title_desc_tags(cfg, topic, script2)

    print("→ Thumbnail…")
    make_thumbnail(title, cfg)

    # Upload
    print("→ Uploading to YouTube…")
    try:
        from youtube_uploader import upload_video
        privacy = cfg.get("privacy_status", "public")
        category = cfg.get("category_id", "19")
        vid = upload_video(
            video_path=str(VIDEO_MP4),
            title=title,
            description=description,
            tags=tags,
            privacy_status=privacy,
            category_id=category,
            thumbnail_path=str(THUMB)
        )
        print("Uploaded video id:", vid)
    except Exception as e:
        print("Upload failed:", e)
        print("Video, audio, and assets are in ./out for manual upload.")

if __name__ == "__main__":
    run()
