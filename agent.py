import os, json, math, time, base64, io, pathlib, random, re, shutil
from datetime import datetime
from typing import List

import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from openai import OpenAI
from num2words import num2words

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
client = OpenAI()

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
def normalize_numbers_for_voice(text: str) -> str:
    """
    Convert numerals to words for narration:
    - Years: pair-speak (e.g., 1665 -> 'sixteen sixty five'; 2025 -> 'twenty twenty five'; 1905 -> 'nineteen oh five'; 1900 -> 'nineteen hundred'; 2000 -> 'two thousand').
    - Currency: $/€/£/¥/₹/฿ -> '... dollars/euros/pounds/yen/rupees/baht' (uses 'point' for decimals).
    - Percentages: '12%' -> 'twelve percent'
    - General numbers: 1234 -> 'one thousand two hundred thirty four'
    """
    import re
from num2words import num2words

    # --- helpers ---
    def num_words(n_str: str) -> str:
        # supports commas/decimals
        s = n_str.replace(",", "")
        if "." in s:
            whole, frac = s.split(".", 1)
            base = num2words(int(whole))
            # spell decimal as "point five six"
            frac_words = " ".join(num2words(int(d)) for d in frac if d.isdigit())
            return f"{base} point {frac_words}" if frac_words else base
        return num2words(int(s))

    def year_to_words(y: int) -> str:
        if y == 2000:
            return "two thousand"
        if 2010 <= y <= 2099:
            last = y % 100
            last_words = " ".join(num2words(last).replace("-", " ").split())
            return f"twenty {last_words}"
        if 2001 <= y <= 2009:
            last = y % 100
            last_words = " ".join(num2words(last).replace("-", " ").split())
            return f"two thousand {last_words}"
        if 1900 <= y <= 1999:
            first = "nineteen"
            last = y % 100
            if last == 0:
                return f"{first} hundred"
            if last < 10:
                return f"{first} oh {num2words(last)}"
            return f"{first} " + " ".join(num2words(last).replace("-", " ").split())
        if 1000 <= y <= 1899:
            first = num2words(y // 100).replace("-", " ")
            last = y % 100
            if last == 0:
                return f"{first} hundred"
            if last < 10:
                return f"{first} oh {num2words(last)}"
            return f"{first} " + " ".join(num2words(last).replace("-", " ").split())
        # fallback
        return " ".join(num2words(y).replace("-", " ").split())

    currency_units = {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen", "₹": "rupees", "฿": "baht"}

    # --- 1) currency amounts ---
    def _cur_repl(m):
        sym = m.group("sym")
        num = m.group("num")
        unit = currency_units.get(sym, "dollars")
        return f"{num_words(num)} {unit}"
    text = re.sub(r'(?P<sym>[$€£¥₹฿])\s?(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?)', _cur_repl, text)

    # --- 2) percentages ---
    def _pct_repl(m):
        val = m.group("num")
        return f"{num_words(val)} percent"
    text = re.sub(r'(?P<num>\d+(?:\.\d+)?)\s?%', _pct_repl, text)

    # --- 3) years (four digits) ---
    def _year_repl(m):
        y = int(m.group(0))
        return year_to_words(y)
    text = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', _year_repl, text)

    # --- 4) general numbers (avoid already-converted pieces) ---
    def _num_repl(m):
        s = m.group(0)
        return " ".join(num_words(s).replace("-", " ").split())
    text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', _num_repl, text)

    # Tidy spaces
    return re.sub(r'\s+', ' ', text).strip()

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
        f"Write a natural, human-sounding narration for a {cfg.get('duration_minutes',10)}-minute YouTube video titled:\n"
        f"{topic}\n\n"
        "Hook in 10s, clear structure, concrete prices, crisp sentences. Output narration only."
    )
    return chat(cfg["writer_model"], sys, ask)

def critique_and_revise(cfg, script: str) -> str:
    sys = system_critic()
    ask = "Improve clarity, pacing, and retention cues. Keep length. Return revised narration only.\n\n" + script
    return chat(cfg["critic_model"], sys, ask)

def extract_broll_keywords(cfg, script: str) -> list[str]:
    """
    Produce strong, chapter-aware b-roll keywords while filtering junk.
    Returns a flat, de-duplicated list so the rest of the pipeline stays the same.
    Also writes a structured plan to out/keywords.json for inspection.
    """
    import json, re

    # Terms we never want (no more random cabbage or clip-arty stuff)
    BLOCKLIST = {
        "cabbage","lettuce","broccoli","cauliflower","salad","cartoon","clipart","vector",
        "logo","pattern","wallpaper","abstract texture","template","comic","emoji","infographic"
    }

    # 1) Ask the model for a structured, chapter-wise plan (JSON)
    sys = (
        "You are a travel-video b-roll planner. Read the script and return compact JSON with:\n"
        "{ \"global\": [10 concise keywords],\n"
        "  \"chapters\": [ {\"title\": \"...\", \"keywords\": [10-12 concise keywords]} ] }\n"
        "Use concrete nouns and scenes that match travel + MONEY when relevant "
        "(e.g., cash close-up, currency exchange board, market price signs, ATM withdrawal, "
        "credit card tap, budget accommodation room, bus/metro/plane). "
        "No vague adjectives. No food unless the script explicitly mentions it. "
        "Keywords should be short search phrases (2–4 words)."
    )
    txt = chat(cfg["critic_model"], sys, script)

    plan = None
    # try to parse JSON directly; if the model wrapped it in prose, extract the JSON blob
    try:
        plan = json.loads(txt)
    except Exception:
        m = re.search(r'\{.*\}', txt, flags=re.S)
        if m:
            try:
                plan = json.loads(m.group(0))
            except Exception:
                plan = None

    keywords: list[str] = []
    structured = {"global": [], "chapters": []}

    if isinstance(plan, dict):
        g = plan.get("global", []) or []
        keywords.extend(g)
        for ch in plan.get("chapters", []) or []:
            ks = ch.get("keywords", []) or []
            structured["chapters"].append({"title": ch.get("title", ""), "keywords": ks})
            keywords.extend(ks)
        structured["global"] = g

    # 2) Fallback if JSON failed: simple list like before, but tell it to favor money + location
    if not keywords:
        ask = (
            "List 40 short, comma-separated b-roll search keywords (no numbering). "
            "Favor money/finance visuals when the script references prices, budgets, cards, or cash; "
            "otherwise prefer specific locations and travel scenes mentioned."
        )
        resp = chat(cfg["critic_model"], "You generate concise keyword lists.", ask + "\n\n" + script)
        keywords = re.split(r"[,\n]", resp)

    # 3) Clean, blocklist, and de-duplicate
    cleaned: list[str] = []
    seen = set()
    for k in keywords:
        k = (k or "").strip()
        if not k:
            continue
        k = re.sub(r"[^a-zA-Z0-9\s\-'/,]+", "", k)           # drop odd chars
        k = re.sub(r"\s+", " ", k).strip()
        low = k.lower()
        if low in seen:
            continue
        if any(b in low for b in BLOCKLIST):
            continue
        # discard overly generic single words
        if len(k.split()) == 1 and k.lower() in {
            "travel","city","street","building","nature","photo","landscape","people"
        }:
            continue
        seen.add(low)
        cleaned.append(k)

    # 4) If the script talks about money, force-boost money visuals to the front
    if re.search(r"\b(dollar|euro|peso|baht|yen|rupee|pound|card|cash|budget|price|cost|exchange|atm)\b",
                 script, flags=re.I):
        money_boost = [
            "cash close up","counting money hands","credit card tap",
            "currency exchange board","atm withdrawal","market price signs"
        ]
        for term in reversed(money_boost):  # insert at front preserving order
            if term not in cleaned:
                cleaned.insert(0, term)

    # 5) Trim to a sensible maximum
    cleaned = cleaned[:120]

    # 6) Save the structured plan for review (optional)
    try:
        (OUT / "keywords.json").write_text(
            json.dumps(structured, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

    return cleaned
(cfg, script: str) -> List[str]:
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
        total = max(10.0, audio.duration)
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
        prompt = f"Bold, high-contrast travel thumbnail background, no text. Topic: {title}"
        r = client.images.generate(model=cfg.get("image_model", "gpt-image-1"), prompt=prompt, size="1536x1024")
        b64 = r.data[0].b64_json
        Image.open(io.BytesIO(base64.b64decode(b64))).save(THUMB, "PNG")
    except Exception as e:
        print(f"Thumbnail gen failed: {e}")
        fallback_background(title, 1280, 720).save(THUMB, "PNG")
    return THUMB

# ---------- Optional YouTube upload ----------
def maybe_upload_to_youtube(cfg: dict, title: str, description: str, video_path: pathlib.Path, thumb_path: pathlib.Path):
    if not (YT_CLIENT_ID and YT_CLIENT_SECRET and YT_REFRESH_TOKEN):
        print("YouTube secrets not set; skipping upload.")
        return
    try:
        from youtube_uploader import upload_video
       upload_video(
    str(video_path),
    title,
    description,
    str(thumb_path) if thumb_path.exists() else None,
    cfg.get("privacy_status", "public"),
    str(cfg.get("category_id", "19")),
    YT_CLIENT_ID,
    YT_CLIENT_SECRET,
    YT_REFRESH_TOKEN,
)
    except Exception as e:
        print(f"YouTube upload skipped: {e}")

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
    elevenlabs_tts(cfg, normalize_numbers_for_voice(script2), VOICE_MP3)

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
