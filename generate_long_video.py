"""
Long-form Omni Mystery video: deep-dive listicle (8-12 min, 5-7 facts).
Pipeline: LLM generates deep-dive script → edge-tts → Pexels clips → ffmpeg → upload
"""

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import edge_tts
import requests

# ── Constants ──────────────────────────────────────────────────────────
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_PATH = BUILD_DIR / "voiceover.mp3"
MUSIC_PATH = BUILD_DIR / "music.mp3"
OUTPUT_PATH = BUILD_DIR / "output_mystery_long.mp4"
HISTORY_PATH = Path("topic_history_long.json")
MAX_HISTORY = 30

TARGET_W, TARGET_H = 1280, 720
FPS = 30
FFMPEG_PRESET = "medium"
FFMPEG_CRF = "23"

TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-AndrewMultilingualNeural",
]
TTS_RATE_OPTIONS = ["+3%", "+5%", "+7%"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

DEEP_DIVE_TOPICS = [
    # Space deep-dives
    "The Fermi Paradox: Why Haven't We Found Aliens Yet",
    "Black Holes: What Happens If You Fall Into One",
    "The Voyager Mission: Humanity's Farthest Travelers",
    "Dark Matter: The Invisible 85% of the Universe",
    "The Multiverse Theory: Could Parallel Universes Exist",
    "Gamma Ray Bursts: The Most Powerful Explosions in Space",
    "The Dyson Sphere: How an Alien Civilization Could Harvest a Star",
    "Rogue Planets: Worlds Without a Sun",
    "The Heat Death of the Universe: How Everything Ends",
    "Time Dilation: How Gravity Bends Time Itself",
    "The Observable Universe: 93 Billion Light Years Across",
    "Neutron Stars: A Teaspoon Weighs a Billion Tons",
    "The Kardashev Scale: Measuring Alien Civilizations",
    "Oumuamua: The Mysterious Interstellar Visitor",
    "The Great Silence: Why the Universe Is So Quiet",
    # Ocean deep-dives
    "The Mariana Trench: Deepest Place on Earth",
    "Bioluminescence: The Glow-in-the-Dark Ocean",
    "Deep Sea Hydrothermal Vents: Life Without Sunlight",
    "Giant Squid: Real Sea Monsters of the Deep",
    "The Bermuda Triangle: Science Behind the Mystery",
    "Underwater Rivers: Rivers Flowing Inside the Ocean",
    # Mystery deep-dives
    "The Wow Signal: 72 Seconds From Beyond",
    "The Dyatlov Pass Incident: What Really Happened",
    "The Voynich Manuscript: A Book Nobody Can Read",
    "Gobekli Tepe: A Temple Older Than Civilization",
    "The Tunguska Event: The Explosion That Flattened 2000 Square Kilometers",
    "The Antikythera Mechanism: An Ancient Computer",
    "DB Cooper: The Only Unsolved Hijacking in History",
    "The Eye of the Sahara: Earth's Bullseye",
    "Stonehenge: How Was It Really Built",
    # Science deep-dives
    "The Double Slit Experiment: When Physics Got Weird",
    "Tardigrades: The Indestructible Micro-Animals",
    "CRISPR: Rewriting the Code of Life",
    "The Mpemba Effect: Why Hot Water Freezes Faster",
    "Quantum Tunneling: Walking Through Walls",
    "Antimatter: The Most Expensive Substance on Earth",
    "The Doomsday Vault: Saving Seeds for the Apocalypse",
    "Supervolcano Yellowstone: The Ticking Time Bomb",
]

PEXELS_QUERIES = [
    "dark space galaxy nebula", "planet earth orbit night",
    "stars universe deep field", "astronaut space station",
    "black hole visualization", "nebula colorful space",
    "ocean deep underwater dark", "underwater deep sea creatures",
    "jellyfish underwater glowing", "bioluminescent ocean",
    "ancient ruins mystery dark", "foggy forest mystery",
    "abandoned building dark", "cave underground explore",
    "pyramid egypt ancient", "stone circle monument",
    "aurora borealis northern lights", "volcano eruption lava",
    "lightning storm dark sky", "ice glacier landscape",
    "laboratory science dark", "circuit board technology",
    "radar screen technology", "microscope science closeup",
    "desert night sky stars", "waterfall jungle mist",
    "telescope observatory", "coral reef underwater dark",
    "supernova star explosion", "rocket launch space",
]

MUSIC_URLS = [
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
]

_DESCRIPTION_FOOTER = (
    "\n\n---\n"
    "Subscribe for deep dives into the universe's biggest mysteries! 🔔\n"
    "What topic should we explore next? Comment below 👇\n\n"
    "#space #mystery #science #deepdive #universe"
)

_CORE_TAGS = [
    "space", "mystery", "science", "deepdive", "universe",
    "facts", "documentary", "educational",
]

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"


# ── Helpers ────────────────────────────────────────────────────────────
def _clean_build():
    if BUILD_DIR.is_dir():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg(cmd: list):
    print(f"[CMD] {' '.join(cmd[:8])}... ({len(cmd)} args)")
    subprocess.run(cmd, check=True)


def _probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def _groq_call(messages: list, temperature: float = 0.8, max_tokens: int = 8192) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    for attempt in range(1, 3):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[WARN] Groq attempt {attempt}: {exc}")
            time.sleep(5)
    return None


def _load_history() -> list[str]:
    if HISTORY_PATH.is_file():
        try:
            return json.loads(HISTORY_PATH.read_text("utf-8"))
        except Exception:
            pass
    return []


def _save_history(h: list[str]):
    HISTORY_PATH.write_text(json.dumps(h[-MAX_HISTORY:], ensure_ascii=False), encoding="utf-8")


def _pick_topic() -> str:
    history = _load_history()
    available = [t for t in DEEP_DIVE_TOPICS if t not in history]
    if not available:
        available = list(DEEP_DIVE_TOPICS)
        history.clear()
    try:
        from analytics import get_topic_weights
        weights = get_topic_weights(available)
        topic = random.choices(available, weights=weights, k=1)[0] if weights else random.choice(available)
    except Exception:
        topic = random.choice(available)
    history.append(topic)
    _save_history(history)
    return topic


# ── Script Generation ────────────────────────────────────────────────
def generate_deep_dive_script(topic: str, min_words: int = 500) -> Optional[dict]:
    fact_count = random.choice([5, 6, 7])

    messages = [
        {"role": "system", "content": (
            "You are a science documentary narrator for YouTube. "
            "Your style: authoritative yet accessible, like a Netflix documentary.\n\n"
            "RULES:\n"
            "- Each sentence: max 15 words (for TTS).\n"
            "- Include specific measurements, dates, numbers, names.\n"
            "- Use vivid comparisons: 'If you compressed Earth to the size of a marble...'\n"
            "- NO generic filler: 'you won't believe', 'this is amazing', 'trust me'.\n"
            "- Build gradually from known to mind-blowing.\n"
            "- Respond with ONLY a JSON object. No markdown, no commentary.\n"
        )},
        {"role": "user", "content": f"""Write a deep-dive YouTube video script: "{topic}"

This is an 8-12 minute video. The script MUST be 1500-2000 words long.

CRITICAL: The "script" field must contain AT LEAST 1500 words. This is a LONG video, not a short.
If the script is under 1000 words, the video cannot be produced.

STRUCTURE (write ALL of these sections in full):
1. HOOK (3-4 sentences): Start with the most mind-blowing fact about this topic.
2. CONTEXT (2-3 sentences): Brief background — why should anyone care?
3. DEEP DIVE ({fact_count} sections, each 200-250 words):
   - Section 1: The basics that most people get wrong
   - Section 2: The first surprising discovery
   - Section 3: The hidden connection nobody talks about
   - Section 4: The mind-blowing measurement or comparison
   - Section 5: The implications that change everything
   {"- Section 6: The latest breakthrough or mystery" if fact_count >= 6 else ""}
   {"- Section 7: What scientists predict will happen next" if fact_count >= 7 else ""}
   - Each section: include specific numbers, dates, names, distances
   - Use analogies to make abstract concepts tangible
   - Transition between sections naturally
4. IMPLICATIONS (3-4 sentences): What does this mean for humanity?
5. OUTRO (2-3 sentences): Biggest takeaway + subscribe CTA.

Return a JSON object with these exact keys:
- "title": string, engaging title max 90 chars with emoji
- "description": string, 5-8 lines with hashtags
- "tags": array of 15-20 strings
- "pexels_queries": array of 6-8 English search queries for footage
- "script": ONE STRING with the full narration (1500-2000 words), sentences separated by newlines. NOT an array."""},
    ]

    content = _groq_call(messages, temperature=0.85, max_tokens=8192)
    if not content:
        return None
    try:
        data = json.loads(content)
        script = data.get("script", "")
        if isinstance(script, list):
            script = "\n".join(str(s) for s in script)
            data["script"] = script
        wc = len(script.split())
        print(f"[SCRIPT] {wc} words, {fact_count} sections, topic: {topic}")
        if wc < min_words:
            return None
        return data
    except Exception as exc:
        print(f"[WARN] Parse: {exc}")
        return None


# ── TTS ───────────────────────────────────────────────────────────────
async def _generate_tts(text: str, path: Path) -> list[dict]:
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    comm = edge_tts.Communicate(text, voice, rate=rate)
    events = []
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                events.append({
                    "text": chunk["text"],
                    "offset": chunk["offset"] / 10_000_000,
                    "duration": chunk["duration"] / 10_000_000,
                })
    print(f"[TTS] {voice} rate={rate}, {len(events)} words")
    return events


def generate_tts(text: str) -> tuple[Path, list[dict]]:
    return AUDIO_PATH, asyncio.run(_generate_tts(text, AUDIO_PATH))


# ── Clips ─────────────────────────────────────────────────────────────
def _download_file(url: str, dest: Path):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_clips(extra: list[str] = None, target: int = 35) -> list[Path]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    queries = list(extra or [])
    base = [q for q in PEXELS_QUERIES if q not in queries]
    random.shuffle(base)
    queries.extend(base)

    headers = {"Authorization": api_key}
    paths, seen = [], set()
    idx = 0
    for query in queries:
        if len(paths) >= target:
            break
        try:
            resp = requests.get("https://api.pexels.com/videos/search", headers=headers,
                                params={"query": query, "per_page": 3, "orientation": "landscape"}, timeout=30)
            resp.raise_for_status()
        except Exception:
            continue
        for video in resp.json().get("videos", []):
            vid_id = video.get("id")
            if vid_id in seen:
                continue
            seen.add(vid_id)
            hd = [f for f in video.get("video_files", []) if (f.get("height") or 0) >= 720]
            if not hd:
                continue
            best = min(hd, key=lambda f: abs((f.get("height") or 0) - 720))
            idx += 1
            p = CLIPS_DIR / f"clip_{idx:03d}.mp4"
            try:
                _download_file(best["link"], p)
                paths.append(p)
            except Exception:
                pass
            if len(paths) >= target:
                break
    print(f"[CLIPS] {len(paths)} downloaded")
    return paths


def download_music() -> Optional[Path]:
    for url in random.sample(MUSIC_URLS, len(MUSIC_URLS)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    return None


# ── FFmpeg Assembly ──────────────────────────────────────────────────
def _prepare_clip(src: Path, dst: Path, duration: int = 5):
    vf = f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,crop={TARGET_W}:{TARGET_H},fps={FPS}"
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(src), "-t", str(duration), "-vf", vf,
                 "-an", "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, str(dst)])


def _fmt_ass_time(s: float) -> str:
    cs = max(0, int(round(s * 100)))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def _safe_text(raw: str) -> str:
    t = raw.replace("\\", " ").replace("\n", " ").replace(":", " ").replace(";", " ")
    t = t.replace("'", "").replace('"', "")
    return re.sub(r"\s+", " ", t).strip() or " "


def _group_words(events: list[dict], max_per: int = 5) -> list[dict]:
    if not events:
        return []
    lines, buf, start, end, kara = [], [], 0.0, 0.0, []
    for ev in events:
        s, d = ev["offset"], ev["duration"]
        if buf and (len(buf) >= max_per or (s - end) > 0.6):
            lines.append({"start": start, "end": end, "text": " ".join(buf), "words": list(kara)})
            buf, kara = [], []
        if not buf:
            start = s
        buf.append(ev["text"])
        kara.append({"text": ev["text"], "offset": s, "duration": d})
        end = s + d
    if buf:
        lines.append({"start": start, "end": end, "text": " ".join(buf), "words": list(kara)})
    return lines


def _write_ass(word_events: list[dict], ass_path: Path) -> Path:
    # Cinematic red-to-white karaoke for mystery/science tone
    primary = "&H000040FF"     # Red-orange (spoken)
    secondary = "&H00FFFFFF"   # White (upcoming)
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
        f"PlayResX: {TARGET_W}\nPlayResY: {TARGET_H}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Kara,DejaVu Sans,44,{primary},{secondary},&H00000000,&H80000000,"
        "1,0,0,0,100,100,1,0,1,3,2,2,30,30,80,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = _group_words(word_events)
    events = []
    for line in lines:
        s, e = line["start"], line["end"] + 0.15
        parts = [f"{{\\kf{max(5, int(w['duration'] * 100))}}}{_safe_text(w['text']).upper()}" for w in line["words"]]
        events.append(f"Dialogue: 0,{_fmt_ass_time(s)},{_fmt_ass_time(e)},Kara,,0,0,0,,{' '.join(parts)}")
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[SUBS] {len(events)} lines → {ass_path}")
    return ass_path


def assemble_video(clips: list[Path], voiceover: Path, word_events: list[dict], music: Optional[Path]) -> Path:
    temp = BUILD_DIR / "temp"
    temp.mkdir(exist_ok=True)

    prepared = []
    for i, clip in enumerate(clips):
        dst = temp / f"prep_{i:03d}.mp4"
        _prepare_clip(clip, dst, duration=5)
        prepared.append(dst)

    concat_file = temp / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in prepared), encoding="utf-8")
    silent = temp / "silent.mp4"
    _run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(silent)])

    voice_dur = _probe_duration(voiceover)
    clip_dur = _probe_duration(silent)
    final_dur = voice_dur + 1.5

    if clip_dur < voice_dur:
        looped = temp / "looped.mp4"
        _run_ffmpeg(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(silent),
                     "-t", f"{final_dur:.2f}", "-c", "copy", str(looped)])
        silent = looped

    ass_path = _write_ass(word_events, temp / "captions.ass")
    graded = temp / "graded.mp4"
    ass_esc = ass_path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("[", "\\[").replace("]", "\\]")

    # Cinematic color grading: slightly desaturated, darker — fits mystery/space theme
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(silent),
                 "-vf", f"eq=contrast=1.1:brightness=-0.03:saturation=0.85,subtitles={ass_esc}",
                 "-t", f"{final_dur:.2f}", "-c:v", "libx264", "-preset", FFMPEG_PRESET,
                 "-crf", FFMPEG_CRF, "-an", str(graded)])

    voice_pad = f"apad=whole_dur={final_dur:.2f}"
    cmd = ["ffmpeg", "-y", "-i", str(graded), "-i", str(voiceover)]
    if music and music.exists():
        cmd.extend(["-stream_loop", "-1", "-i", str(music)])
        cmd.extend(["-filter_complex",
                     (f"[1:a]acompressor=threshold=-18dB:ratio=2.5:attack=5:release=120,{voice_pad}[va];"
                      "[va]asplit=2[va1][va2];"
                      "[2:a]highpass=f=80,lowpass=f=14000,volume=0.12[ma];"
                      "[ma][va1]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=250[ducked];"
                      "[va2][ducked]amix=inputs=2:duration=first:normalize=0[a]"),
                     "-map", "0:v", "-map", "[a]"])
    else:
        cmd.extend(["-filter_complex", f"[1:a]{voice_pad}[a]", "-map", "0:v", "-map", "[a]"])
    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-t", f"{final_dur:.2f}", "-movflags", "+faststart", str(OUTPUT_PATH)])
    _run_ffmpeg(cmd)
    print(f"[VIDEO] voice={voice_dur:.1f}s final={final_dur:.1f}s → {OUTPUT_PATH}")
    return OUTPUT_PATH


# ── Upload ────────────────────────────────────────────────────────────
def _get_access_token() -> str:
    resp = requests.post(TOKEN_URL, data={
        "client_id": os.environ["YT_CLIENT_ID"],
        "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload_video(meta: dict) -> str:
    creds = [os.getenv("YT_CLIENT_ID"), os.getenv("YT_CLIENT_SECRET"), os.getenv("YT_REFRESH_TOKEN")]
    if not all(creds):
        print("[SKIP] Upload: missing credentials")
        return ""
    if not OUTPUT_PATH.is_file():
        return ""

    privacy = os.getenv("YOUTUBE_PRIVACY", "public")
    if privacy not in ("public", "unlisted", "private"):
        privacy = "public"

    access_token = _get_access_token()
    body = {
        "snippet": {
            "title": meta.get("title", "Deep Dive: Space Mystery")[:100],
            "description": meta.get("description", ""),
            "tags": meta.get("tags", _CORE_TAGS),
            "categoryId": "28",
            "defaultLanguage": "en",
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False, "embeddable": True},
    }

    video_data = OUTPUT_PATH.read_bytes()
    init_resp = requests.post(UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
                              headers={"Authorization": f"Bearer {access_token}",
                                       "Content-Type": "application/json; charset=UTF-8",
                                       "X-Upload-Content-Length": str(len(video_data)),
                                       "X-Upload-Content-Type": "video/mp4"},
                              json=body, timeout=30)
    init_resp.raise_for_status()
    upload_url = init_resp.headers["Location"]

    print(f"[UPLOAD] {len(video_data) / 1024 / 1024:.1f} MB...")
    for attempt in range(1, 4):
        try:
            resp = requests.put(upload_url, headers={"Authorization": f"Bearer {access_token}",
                                                      "Content-Type": "video/mp4",
                                                      "Content-Length": str(len(video_data))},
                                data=video_data, timeout=600)
            resp.raise_for_status()
            video_id = resp.json().get("id", "")
            print(f"[UPLOAD] https://youtube.com/watch?v={video_id}")
            try:
                from analytics import log_upload
                log_upload(video_id, meta.get("title", ""), meta.get("topic", ""), meta.get("tags", []))
            except Exception as exc:
                print(f"[WARN] Analytics: {exc}")
            return video_id
        except Exception as exc:
            print(f"[WARN] Upload attempt {attempt}: {exc}")
            if attempt < 3:
                time.sleep(attempt * 15)
    return ""


# ── Main ─────────────────────────────────────────────────────────────
def main():
    _clean_build()

    print("[1/5] Generating deep-dive script...")
    topic = _pick_topic()
    print(f"  Topic: {topic}")

    script_data = None
    # Progressive retry: first try needs 500 words, subsequent tries relax to 300
    for attempt in range(4):
        min_w = 500 if attempt < 2 else 300
        script_data = generate_deep_dive_script(topic, min_words=min_w)
        if script_data:
            break
        print(f"[RETRY] Attempt {attempt + 2} (min_words={min_w})...")
    if not script_data:
        print("[ERROR] Failed to generate script")
        sys.exit(1)

    script_text = script_data["script"]
    if isinstance(script_text, list):
        script_text = "\n".join(str(s) for s in script_text)
    meta = {
        "title": script_data.get("title", topic)[:100],
        "description": script_data.get("description", "") + _DESCRIPTION_FOOTER,
        "tags": list(dict.fromkeys(script_data.get("tags", []) + _CORE_TAGS))[:20],
        "topic": topic,
    }
    print(f"  Title: {meta['title']}")
    print(f"  Script: {len(script_text.split())} words")

    (BUILD_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[2/5] Generating voiceover...")
    audio_path, word_events = generate_tts(script_text)
    dur = _probe_duration(audio_path)
    print(f"  Duration: {dur:.1f}s ({dur/60:.1f} min)")

    print("[3/5] Downloading clips...")
    clips = download_clips(extra=script_data.get("pexels_queries", []), target=35)
    if not clips:
        print("[ERROR] No clips")
        sys.exit(1)

    print("[4/5] Downloading music...")
    music = download_music()

    print("[5/5] Assembling video...")
    assemble_video(clips, audio_path, word_events, music)

    print("[UPLOAD] Uploading...")
    upload_video(meta)

    temp = BUILD_DIR / "temp"
    if temp.is_dir():
        shutil.rmtree(temp)
    print("[DONE]")


if __name__ == "__main__":
    main()
