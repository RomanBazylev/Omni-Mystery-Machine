import os
import asyncio
import json
import random
import re
import shutil
import requests
from pathlib import Path
from groq import Groq
import edge_tts
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
)

from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from youtube_uploader import upload_to_youtube

# ── Config ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
BUILD_DIR = Path("build")

# ── Hardcoded Pexels queries — never let LLM choose ───────────────
PEXELS_QUERIES = [
    "dark space galaxy nebula",
    "planet earth orbit night",
    "stars universe deep field",
    "astronaut space station dark",
    "moon surface crater closeup",
    "milky way timelapse night sky",
    "aurora borealis northern lights",
    "rocket launch space night",
    "satellite orbit earth dark",
    "solar system planets dark",
    "ocean deep underwater dark",
    "lightning storm dark sky",
    "ancient ruins mystery dark",
    "foggy forest mystery dark",
    "abandoned building dark mystery",
    "cave underground dark explore",
    "volcano eruption night lava",
    "desert night sky stars",
    "underwater deep sea creatures",
    "pyramid egypt ancient night",
    "ice glacier frozen landscape",
    "dark tunnel underground mystery",
    "meteor shower night sky",
    "nebula colorful space deep",
    "mars red planet surface",
    "saturn rings planet space",
    "comet tail space dark",
    "jupiter clouds planet space",
    "telescope observatory night",
    "supernova star explosion space",
]

# ── Pixabay queries (backup source) ────────────────────────────────
PIXABAY_QUERIES = [
    "space galaxy stars",
    "planet earth night",
    "nebula universe cosmos",
    "deep ocean underwater",
    "lightning storm nature",
    "volcano lava eruption",
    "ancient ruins temple",
    "foggy forest mystery",
]

# ── Blacklist — skip clips whose tags contain these words ──────────
_BLACKLIST_WORDS = {
    "meeting", "teamwork", "handshake", "presentation",
    "conference", "hug", "embrace", "couple", "friends",
    "love", "together", "celebrate", "portrait", "face",
    "smile", "happy", "group", "crowd", "party", "wedding",
    "romantic", "family", "children", "kid", "fashion",
    "model", "beauty", "lifestyle", "yoga", "fitness",
    "dance", "selfie", "corporate", "suit", "interview",
    "office", "business", "classroom", "student",
}


# ── Helpers ─────────────────────────────────────────────────────────

def _clean_build() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def _pexels_best_file(video_files: list) -> dict | None:
    """Pick the best HD portrait file from Pexels video_files list."""
    hd = [f for f in video_files if (f.get("height") or 0) >= 720]
    if hd:
        return min(hd, key=lambda f: abs((f.get("height") or 0) - 1920))
    if video_files:
        return max(video_files, key=lambda f: f.get("height") or 0)
    return None


# ── LLM ─────────────────────────────────────────────────────────────

def get_content() -> dict:
    """Generate a space/mystery fact via Groq with retry + fallback."""
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "You are a viral YouTube Shorts scriptwriter for a space & mystery channel "
        "called 'Void Chronicles AI'.\n"
        "Generate ONE mind-blowing fact about space, deep ocean, unsolved mysteries, "
        "or conspiracy theories.\n\n"
        "REQUIREMENTS:\n"
        "- text: 2-4 sentences, 40-80 words. Shocking, specific, real numbers/names.\n"
        "- title: Catchy YouTube title with emoji, max 70 chars, include #Shorts\n"
        "- tags: comma-separated, 6-10 relevant tags\n"
        "- description: 2-3 lines with hashtags\n\n"
        "Return ONLY valid JSON:\n"
        '{"text": "...", "title": "... #Shorts", '
        '"tags": "space, mystery, ...", "description": "... #shorts #space"}'
    )

    for attempt in range(2):
        try:
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.9 if attempt == 0 else 1.0,
            )
            raw = chat.choices[0].message.content
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)

            if data.get("text") and data.get("title"):
                data.setdefault("tags", "space, mystery, facts, shorts, science, ai")
                data.setdefault(
                    "description",
                    f"{data['text']} #shorts #mystery #space",
                )
                return data
            print(f"[WARN] LLM missing fields (attempt {attempt + 1})")
        except Exception as exc:
            print(f"[WARN] Groq error (attempt {attempt + 1}): {exc}")

    # Fallback — hardcoded fact so the run never crashes
    return {
        "text": (
            "The largest known structure in the universe is the "
            "Hercules-Corona Borealis Great Wall. It is so massive that "
            "light takes 10 billion years to travel across it. Our entire "
            "galaxy is just a speck compared to it."
        ),
        "title": "🌌 The Universe Is TERRIFYINGLY Big #Shorts",
        "tags": "space, universe, mystery, facts, shorts, science",
        "description": (
            "The largest structure in the universe will blow your mind 🌌\n"
            "#shorts #space #mystery #facts #science"
        ),
    }


# ── Download helper ─────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


# ── Pexels ──────────────────────────────────────────────────────────

def _download_pexels() -> Path | None:
    """Try Pexels first. Returns path or None."""
    if not PEXELS_API_KEY:
        return None

    headers = {"Authorization": PEXELS_API_KEY}
    queries = list(PEXELS_QUERIES)
    random.shuffle(queries)

    for query in queries[:10]:
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": query, "per_page": 3, "orientation": "portrait"},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Pexels search '{query}' failed: {exc}")
            continue

        for video in resp.json().get("videos", []):
            best = _pexels_best_file(video.get("video_files", []))
            if not best:
                continue
            dest = BUILD_DIR / "source.mp4"
            try:
                _download_file(best["link"], dest)
                print(f"  Pexels [{query}] -> source.mp4")
                return dest
            except Exception as exc:
                print(f"[WARN] Pexels download failed: {exc}")
                continue
    return None


# ── Pixabay (fallback) ──────────────────────────────────────────────

def _download_pixabay() -> Path | None:
    """Fallback to Pixabay. Returns path or None."""
    if not PIXABAY_API_KEY:
        return None

    query = random.choice(PIXABAY_QUERIES)
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "per_page": 10,
        "safesearch": "true",
        "order": "popular",
    }
    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        safe_msg = str(exc)
        if PIXABAY_API_KEY:
            safe_msg = safe_msg.replace(PIXABAY_API_KEY, "***")
        print(f"[WARN] Pixabay API error: {safe_msg}")
        return None

    for hit in resp.json().get("hits", []):
        hit_tags = hit.get("tags", "").lower()
        if any(bw in hit_tags for bw in _BLACKLIST_WORDS):
            continue
        videos = hit.get("videos") or {}
        cand = videos.get("large") or videos.get("medium") or videos.get("small")
        if not cand or "url" not in cand:
            continue
        dest = BUILD_DIR / "source.mp4"
        try:
            _download_file(cand["url"], dest)
            print(f"  Pixabay [{query}] -> source.mp4")
            return dest
        except Exception as exc:
            print(f"[WARN] Pixabay download failed: {exc}")
            continue
    return None


# ── Combined video getter ───────────────────────────────────────────

def get_video() -> Path:
    """Try Pexels, then Pixabay. Raises if both fail."""
    path = _download_pexels()
    if path:
        return path
    print("  Pexels failed, trying Pixabay...")
    path = _download_pixabay()
    if path:
        return path
    raise RuntimeError(
        "Could not download any video. Check PEXELS_API_KEY / PIXABAY_API_KEY."
    )


# ── TTS ─────────────────────────────────────────────────────────────

async def get_audio(text: str) -> Path:
    path = BUILD_DIR / "voice.mp3"
    comm = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await comm.save(str(path))
    return path


# ── Video assembly ──────────────────────────────────────────────────

def make_movie(v_path: Path, a_path: Path, text: str) -> Path:
    audio = AudioFileClip(str(a_path))
    video = VideoFileClip(str(v_path))

    # Force 1080×1920 portrait
    video = video.resize(height=1920)
    if video.w > 1080:
        x_center = video.w / 2
        video = video.crop(x1=x_center - 540, x2=x_center + 540)

    # Match video duration to audio (NO speedx — it desyncs audio)
    duration = audio.duration
    if video.duration >= duration:
        clip = video.subclip(0, duration)
    else:
        clip = video.loop(duration=duration)

    # Yellow text with black stroke for readability on any background
    txt = (
        TextClip(
            text,
            fontsize=60,
            color="yellow",
            stroke_color="black",
            stroke_width=2,
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(900, None),
        )
        .set_duration(duration)
        .set_pos("center")
    )

    final = CompositeVideoClip([clip, txt]).set_audio(audio)
    out_path = BUILD_DIR / "out.mp4"
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac")
    return out_path


# ── Pipeline ────────────────────────────────────────────────────────

async def run():
    print("=== Omni Mystery Machine ===")
    _clean_build()

    data = get_content()
    print(f"  Title: {data['title']}")
    print(f"  Text:  {data['text'][:80]}...")

    v = get_video()
    a = await get_audio(data["text"])
    out = make_movie(v, a, data["text"])

    upload_to_youtube(str(out), data["title"], data["description"], data["tags"])
    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(run())
