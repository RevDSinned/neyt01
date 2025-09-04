import os, json, math, time, base64, io, random, textwrap, pathlib
from datetime import datetime
import requests
from openai import OpenAI
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
from youtube_uploader import upload_video  # local module

ROOT = pathlib.Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- helpers ----------
def oai_chat(model, system, user):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

def oai_image(prompt, size="1280x720"):
    img = client.images.generate(model=CONFIG["image_model"], prompt=prompt, size=size)
    b64 = img.data[0].b64_json
    return base64.b64decode(b64)

def tts(text, path):
    # single-file narration
    with client.audio.speech.with_streaming_response.create(
        model=CONFIG["tts_model"], voice=CONFIG["voice"], input=text
    ) as resp:
        resp.stream_to_file(path)

def sec_from_minutes(m): return int(m * 60)

def clean_filename(s):
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c for c in s if c in keep)[:120].strip().replace(" ", "_")

def save_text(name, text):
    (ROOT / "out").mkdir(exist_ok=True)
    p = ROOT / "out" / name
    p.write_text(text, encoding="utf-8")
    return p

def json_block_from_text(txt):
    s = txt.find("{"); e = txt.rfind("}")
    return json.loads(txt[s:e+1])

# ---------- topic & script ----------
def get_topic():
    if CONFIG["topic_mode"] == "manual":
        return CONFIG.get("topic_manual", "Travel costs 101")
    seeds = "\n".join(f"- {s}" for s in CONFIG["topic_seeds"])
    sys = (ROOT / "prompts/topic_generator_system.md").read_text(encoding="utf-8")
    user = f"Channel pillars: travel + money.\nSeeds:\n{seeds}\nAvoid: {', '.join(CONFIG['avoid_terms'])}\nReturn JSON."
    draft = oai_chat(CONFIG["writer_model"], sys, user)
    obj = json_block_from_text(draft)
    return obj

def write_and_critique(topic_obj):
    writer_sys = (ROOT / "prompts/writer_system.md").read_text(encoding="utf-8")
    user = f"Topic: {topic_obj['topic']}\nAngle: {topic_obj.get('angle','')}\nChannel: {CONFIG['channel_name']}"
    draft = oai_chat(CONFIG["writer_model"], writer_sys, user)
    pkg = {
        "title": "",
        "description": "",
        "tags": [],
        "chapters": [],
        "script": draft,
        "broll_keywords": topic_obj.get("broll_keywords", [])
    }
    critic_sys = (ROOT / "prompts/critic_system.md").read_text(encoding="utf-8")
    critic_user = json.dumps(pkg, ensure_ascii=False)
    improved = oai_chat(CONFIG["critic_model"], critic_sys, critic_user)
    try:
        pkg2 = json_block_from_text(improved)
    except Exception:
        pkg2 = pkg
        pkg2["title"] = f"{topic_obj['topic']} — costs & tactics"
        pkg2["description"] = "Travel + money deep dive."
        pkg2["tags"] = ["travel","budget","digital nomad"]
        pkg2["chapters"] = []
    return pkg2

# ---------- stock images ----------
def pexels_images(query, n=8):
    if not PEXELS_API_KEY: return []
    url = "https://api.pexels.com/v1/search"
    r = requests.get(url, headers={"Authorization": PEXELS_API_KEY}, params={"query":query, "per_page": n})
    if r.status_code != 200: return []
    data = r.json()
    return [p["src"]["large"] for p in data.get("photos", [])]

def pixabay_images(query, n=8):
    if not PIXABAY_API_KEY: return []
    url = "https://pixabay.com/api/"
    r = requests.get(url, params={"key": PIXABAY_API_KEY, "q": query, "image_type":"photo", "per_page": n})
    if r.status_code != 200: return []
    data = r.json()
    return [h["largeImageURL"] for h in data.get("hits", [])]

def download(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)

def collect_broll(keywords, target=30):
    out_dir = ROOT / "out" / "broll"
    if out_dir.exists():
        for f in out_dir.glob("*"): f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = []
    for kw in keywords[:10]:
        urls += pexels_images(kw, 4)
        urls += pixabay_images(kw, 4)
        if len(urls) >= target: break
    paths = []
    for i,u in enumerate(urls[:target], start=1):
        p = out_dir / f"img_{i:02d}.jpg"
        try:
            download(u, p); paths.append(p)
        except Exception:
            continue
    return paths

# ---------- video render ----------
def render_video(audio_path, images, title):
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    per = max(5.0, duration / max(1,len(images)))
    clips = []
    for i, img in enumerate(images):
        clip = ImageClip(str(img)).set_duration(per).resize(height=1080).set_position("center")
        clips.append(clip.crossfadein(0.5) if i>0 else clip)
    if not clips:
        bg = Image.new("RGB",(1920,1080),(18,18,18))
        d = ImageDraw.Draw(bg)
        d.text((80,480), title[:48], fill=(240,240,240))
        tmp = ROOT / "out" / "fallback.jpg"
        bg.save(tmp)
        clips = [ImageClip(str(tmp)).set_duration(duration)]
    video = concatenate_videoclips(clips, method="compose").set_audio(audio)
    out_path = ROOT / "out" / f"{clean_filename(title)}.mp4"
    video.write_videofile(str(out_path), fps=30, codec="libx264", audio_codec="aac", threads=4, temp_audiofile=str(ROOT/"out"/"temp-audio.m4a"), remove_temp=True)
    return out_path

# ---------- thumbnail ----------
def make_thumbnail(title, keywords):
    prompt = f"Bold thumbnail for a YouTube video about: {title}. Style: {CONFIG['thumb_style']}. Photographic background, high contrast."
    img_bytes = oai_image(prompt, size="1280x720")
    out = ROOT / "out" / "thumbnail.png"
    out.write_bytes(img_bytes)
    return out

# ---------- main ----------
def main():
    (ROOT / "out").mkdir(exist_ok=True)
    topic_obj = get_topic() if CONFIG["topic_mode"]=="auto" else {"topic": CONFIG["topic_manual"], "angle":"", "broll_keywords":[]}
    pkg = write_and_critique(topic_obj)
    title = pkg.get("title") or topic_obj["topic"]
    script = pkg.get("script","")
    desc = pkg.get("description","")
    tags = pkg.get("tags", [])
    broll_kw = pkg.get("broll_keywords", topic_obj.get("broll_keywords", ["travel","city","airplane","budget"]))
    audio_path = ROOT / "out" / "voice.mp3"
    tts(script, str(audio_path))
    images = collect_broll(broll_kw, target=math.ceil(sec_from_minutes(CONFIG["duration_minutes"]) / 6))
    video_path = render_video(audio_path, images, title)
    thumb_path = make_thumbnail(title, broll_kw)
    meta = {
        "title": title,
        "description": desc,
        "tags": tags if isinstance(tags, list) else [t.strip() for t in str(tags).split(",") if t.strip()],
        "categoryId": CONFIG["category_id"],
        "privacyStatus": CONFIG["privacy_status"]
    }
    (ROOT / "out" / "meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2), encoding="utf-8")
    vid_id = upload_video(
        file_path=str(video_path),
        title=meta["title"],
        description=meta["description"],
        tags=meta["tags"],
        category_id=meta["categoryId"],
        privacy_status=meta["privacyStatus"],
        thumbnail_path=str(thumb_path)
    )
    print("Uploaded video id:", vid_id)

if __name__ == "__main__":
    main()
