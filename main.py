import asyncio
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

import edge_tts
import requests
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    TextClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
    afx,
)

from youtube_uploader import upload_to_youtube

# ── Data classes ────────────────────────────────────────────────────

@dataclass
class ScriptPart:
    text: str

@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: List[str]


# ── Config ─────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_DIR = BUILD_DIR / "audio_parts"
MUSIC_PATH = BUILD_DIR / "music.mp3"

TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-AndrewMultilingualNeural",
]
TTS_RATE_OPTIONS = ["+5%", "+8%", "+10%"]

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

# ── Mystery/space topics for prompt variety ────────────────────────
TOPICS = [
    "black holes and singularities",
    "dark matter and dark energy",
    "the Fermi Paradox",
    "deep ocean unexplored zones",
    "the Bermuda Triangle",
    "ancient lost civilizations",
    "Area 51 and UFO sightings",
    "the Wow signal from space",
    "rogue planets wandering the galaxy",
    "the Great Attractor pulling galaxies",
    "quantum entanglement and teleportation",
    "the multiverse theory",
    "the Dyson Sphere concept",
    "uncontacted tribes on Earth",
    "the Voyager golden record",
    "magnetars and neutron stars",
    "the Tunguska event 1908",
    "the Mariana Trench deepest point",
    "the simulation hypothesis",
    "gamma ray bursts destroying galaxies",
]

ANGLES = [
    "scary and terrifying",
    "mind-blowing revelation",
    "things scientists can't explain",
    "what they don't teach in school",
    "conspiracy theory deep dive",
    "the most dangerous thing in space",
    "facts that will keep you up at night",
    "the unsolved mystery of",
]


# ── Helpers ─────────────────────────────────────────────────────────

def _clean_build() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _pexels_best_file(video_files: list) -> dict | None:
    """Pick the best HD portrait file from Pexels video_files list."""
    hd = [f for f in video_files if (f.get("height") or 0) >= 720]
    if hd:
        return min(hd, key=lambda f: abs((f.get("height") or 0) - 1920))
    if video_files:
        return max(video_files, key=lambda f: f.get("height") or 0)
    return None


# ── LLM — multi-part script ────────────────────────────────────────

def call_groq_for_script() -> tuple:
    """Generate a multi-part mystery/space script via Groq. Returns (parts, metadata)."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    topic = random.choice(TOPICS)
    angle = random.choice(ANGLES)

    system_prompt = (
        "You are an expert scriptwriter for a viral YouTube Shorts channel called "
        "'Void Chronicles AI'. You create TERRIFYING, mind-blowing scripts about space, "
        "deep ocean, unsolved mysteries, and conspiracy theories. "
        "Every phrase must deliver a SPECIFIC fact: real names, dates, numbers, distances. "
        "NEVER write filler like 'This is amazing' or 'You won't believe this'. "
        "Use a dark, cinematic narration tone — like a documentary narrator revealing secrets. "
        "Respond ONLY with valid JSON, no markdown wrappers."
    )

    user_prompt = f"""Write a YouTube Shorts script (45–60 seconds) about mystery/space.

CONTEXT:
- Topic: {topic}
- Angle: {angle}

CONTENT REQUIREMENTS:
1. First phrase — SCROLL-STOPPING hook: a shocking statement with a SPECIFIC NUMBER or NAME.
2. EVERY phrase must contain SPECIFIC facts: distances, dates, names, measurements.
3. NO filler phrases. Banned: "This is amazing", "You won't believe", "Trust me", "Here's the thing".
4. Each phrase = 1–2 sentences, 12–25 words.
5. Include at least ONE specific measurement or comparison (e.g., "100 billion times heavier than our Sun").
6. Final phrase — eerie cliffhanger or call to action.
7. 10–14 parts total (for 45–60 second video).

Format — strictly JSON:
{{
  "title": "Catchy YouTube title with emoji (max 70 chars) #Shorts",
  "description": "YouTube description (2–3 lines) with hashtags",
  "tags": ["space", "mystery", "shorts", ...4-7 more specific tags],
  "parts": [
    {{ "text": "Phrase with specific terrifying fact, 12-25 words" }}
  ]
}}"""

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(2):
        try:
            chat = client.chat.completions.create(**body)
            raw = chat.choices[0].message.content
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)

            parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
            if len(parts) < 6:
                print(f"[WARN] Only {len(parts)} parts (attempt {attempt + 1})")
                body["temperature"] = 1.0
                continue

            metadata = VideoMetadata(
                title=data.get("title", "")[:100] or "Mystery of the Universe #Shorts",
                description=data.get("description", "") or "#shorts #mystery #space",
                tags=data.get("tags", ["space", "mystery", "shorts"]),
            )
            # Ensure #Shorts in title
            if "#shorts" not in metadata.title.lower():
                metadata.title = metadata.title[:90] + " #Shorts"
            return parts, metadata

        except Exception as exc:
            print(f"[WARN] Groq error (attempt {attempt + 1}): {exc}")
            body["temperature"] = 1.0

    # Fallback — hardcoded script so the run never crashes
    return _fallback_script()


def _fallback_script() -> tuple:
    parts = [
        ScriptPart("In 1977, a radio telescope in Ohio received a signal from deep space that lasted exactly 72 seconds."),
        ScriptPart("Astronomer Jerry Ehman circled the data and wrote 'Wow!' in the margin. That signal has never been explained."),
        ScriptPart("It came from the direction of the constellation Sagittarius, 120 light years away."),
        ScriptPart("The signal was 30 times stronger than normal background radiation from space."),
        ScriptPart("Scientists checked every possible natural source — comets, satellites, reflections. Nothing matched."),
        ScriptPart("The frequency was 1420 megahertz — the exact frequency hydrogen atoms emit naturally."),
        ScriptPart("This is the frequency scientists predicted an intelligent civilization would use to communicate."),
        ScriptPart("Despite monitoring that exact spot in space for over 40 years, the signal has never repeated."),
        ScriptPart("Some researchers believe it was a one-time transmission from an alien civilization that has since gone silent."),
        ScriptPart("We may have received humanity's first message from another world and didn't even realize it in time."),
    ]
    metadata = VideoMetadata(
        title="📡 The WOW Signal Still Can't Be Explained #Shorts",
        description=(
            "In 1977, we received a message from space that lasted 72 seconds. "
            "It has never been explained. 📡\n"
            "#shorts #space #mystery #wowsignal #aliens #science"
        ),
        tags=["space", "mystery", "wow signal", "aliens", "shorts", "science", "universe"],
    )
    return parts, metadata


# ── Clip downloads ──────────────────────────────────────────────────

def download_pexels_clips(target_count: int = 14) -> List[Path]:
    """Download clips from Pexels — 1 clip per query."""
    if not PEXELS_API_KEY:
        return []

    headers = {"Authorization": PEXELS_API_KEY}
    queries = list(PEXELS_QUERIES)
    random.shuffle(queries)
    queries = queries[:target_count]
    result_paths: List[Path] = []
    seen_ids: set = set()
    clip_idx = 0

    for query in queries:
        if len(result_paths) >= target_count:
            break
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": query, "per_page": 1, "orientation": "portrait"},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Pexels search '{query}' failed: {exc}")
            continue

        for video in resp.json().get("videos", []):
            vid_id = video.get("id")
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            best = _pexels_best_file(video.get("video_files", []))
            if not best:
                continue
            clip_idx += 1
            clip_path = CLIPS_DIR / f"pexels_{clip_idx}.mp4"
            try:
                _download_file(best["link"], clip_path)
                result_paths.append(clip_path)
                print(f"    Pexels [{query}] -> clip {clip_idx}")
            except Exception as exc:
                print(f"[WARN] Pexels clip {clip_idx} download failed: {exc}")
            if len(result_paths) >= target_count:
                break

    return result_paths


def download_pixabay_clips(max_clips: int = 3) -> List[Path]:
    """Download supplementary clips from Pixabay."""
    if not PIXABAY_API_KEY:
        return []

    query = random.choice(PIXABAY_QUERIES)
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "per_page": max_clips * 3,
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
        return []

    result_paths: List[Path] = []
    clip_idx = 0
    for hit in resp.json().get("hits", []):
        if len(result_paths) >= max_clips:
            break
        hit_tags = hit.get("tags", "").lower()
        if any(bw in hit_tags for bw in _BLACKLIST_WORDS):
            continue
        videos = hit.get("videos") or {}
        cand = videos.get("large") or videos.get("medium") or videos.get("small")
        if not cand or "url" not in cand:
            continue
        clip_idx += 1
        clip_path = CLIPS_DIR / f"pixabay_{clip_idx}.mp4"
        try:
            _download_file(cand["url"], clip_path)
            result_paths.append(clip_path)
            print(f"    Pixabay [{query}] -> clip {clip_idx}")
        except Exception as exc:
            print(f"[WARN] Pixabay clip {clip_idx} download failed: {exc}")

    return result_paths


# ── Background music ────────────────────────────────────────────────

def download_background_music() -> Optional[Path]:
    """Download a dark/ambient CC track for background."""
    if os.getenv("DISABLE_BG_MUSIC") == "1":
        return None
    if MUSIC_PATH.is_file():
        return MUSIC_PATH

    # Dark/ambient Creative Commons tracks
    candidate_urls = [
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Lobo_Loco/Folkish_things/Lobo_Loco_-_01_-_Acoustic_Dreams_ID_1199.mp3",
    ]
    for url in random.sample(candidate_urls, len(candidate_urls)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    print("[WARN] Could not download any background music")
    return None


# ── TTS with word timings (for karaoke) ────────────────────────────

@dataclass
class WordTiming:
    text: str
    offset: float   # seconds from start
    duration: float  # seconds


async def _generate_part_audio(
    text: str, voice: str, rate: str, out_path: Path
) -> List[WordTiming]:
    """Generate TTS audio and capture per-word timestamps."""
    comm = edge_tts.Communicate(text, voice, rate=rate)
    word_timings: List[WordTiming] = []
    audio_chunks = bytearray()

    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio_chunks.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            word_timings.append(WordTiming(
                text=chunk["text"],
                offset=chunk["offset"] / 10_000_000,
                duration=chunk["duration"] / 10_000_000,
            ))

    with out_path.open("wb") as f:
        f.write(audio_chunks)

    return word_timings


async def _generate_all_audio(
    parts: List[ScriptPart],
) -> tuple:
    """Generate per-part TTS audio with word timings. Returns (paths, timings_per_part)."""
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    audio_paths: List[Path] = []
    all_timings: List[List[WordTiming]] = []

    for i, part in enumerate(parts):
        out = AUDIO_DIR / f"part_{i}.mp3"
        audio_paths.append(out)
        timings = await _generate_part_audio(part.text, voice, rate, out)
        all_timings.append(timings)

    return audio_paths, all_timings


def build_tts_per_part(parts: List[ScriptPart]) -> tuple:
    return asyncio.run(_generate_all_audio(parts))


# ── Video assembly helpers ──────────────────────────────────────────

def _fit_clip_to_frame(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Trim/loop clip to duration, crop to 9:16."""
    if clip.duration > duration + 0.5:
        max_start = clip.duration - duration
        start = random.uniform(0, max_start)
        segment = clip.subclip(start, start + duration)
    else:
        segment = clip.fx(vfx.loop, duration=duration)

    margin = 1.10
    src_ratio = segment.w / segment.h
    target_ratio = TARGET_W / TARGET_H
    if src_ratio > target_ratio:
        segment = segment.resize(height=int(TARGET_H * margin))
    else:
        segment = segment.resize(width=int(TARGET_W * margin))

    segment = segment.crop(
        x_center=segment.w / 2, y_center=segment.h / 2,
        width=TARGET_W, height=TARGET_H,
    )
    return segment


def _apply_ken_burns(clip, duration: float):
    """Slow zoom-in or zoom-out for visual dynamics."""
    direction = random.choice(["in", "out"])
    start_scale = 1.0
    end_scale = random.uniform(1.06, 1.12)
    if direction == "out":
        start_scale, end_scale = end_scale, start_scale

    def make_frame(get_frame, t):
        progress = t / max(duration, 0.01)
        scale = start_scale + (end_scale - start_scale) * progress
        frame = get_frame(t)
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        return arr[y_off:y_off + h, x_off:x_off + w]

    return clip.fl(make_frame)


def _make_karaoke_subtitle(
    word_timings: List[WordTiming], duration: float
) -> list:
    """Karaoke-style subtitles: words highlight yellow as they are spoken.

    Shows 3-5 words at a time. Current word is YELLOW, spoken words are WHITE,
    upcoming words are dim gray. Creates the "word-by-word glow" effect.
    """
    if not word_timings:
        return []

    # Group words into chunks of 3-4 for readability
    CHUNK_SIZE = 4
    chunks = []
    for i in range(0, len(word_timings), CHUNK_SIZE):
        chunks.append(word_timings[i:i + CHUNK_SIZE])

    layers = []
    for chunk in chunks:
        chunk_start = chunk[0].offset
        chunk_end = chunk[-1].offset + chunk[-1].duration + 0.1
        chunk_end = min(chunk_end, duration)
        chunk_dur = chunk_end - chunk_start
        if chunk_dur <= 0:
            continue

        # Full chunk text as dim background (upcoming words)
        full_text = " ".join(w.text for w in chunk)
        bg_txt = (
            TextClip(
                full_text,
                fontsize=72,
                color="#888888",
                font="DejaVu-Sans-Bold",
                method="caption",
                size=(TARGET_W - 80, None),
                stroke_color="black",
                stroke_width=4,
            )
            .set_position(("center", 0.75), relative=True)
            .set_start(chunk_start)
            .set_duration(chunk_dur)
        )
        layers.append(bg_txt)

        # Each word highlights yellow when spoken
        for w in chunk:
            w_start = w.offset
            w_end = min(w.offset + w.duration + 0.15, chunk_end)
            w_dur = w_end - w_start
            if w_dur <= 0:
                continue

            highlight = (
                TextClip(
                    w.text,
                    fontsize=78,
                    color="yellow",
                    font="DejaVu-Sans-Bold",
                    method="caption",
                    size=(TARGET_W - 80, None),
                    stroke_color="black",
                    stroke_width=3,
                )
                .set_position(("center", 0.73), relative=True)
                .set_start(w_start)
                .set_duration(w_dur)
            )
            layers.append(highlight)

        # After all words spoken: show full chunk in white
        last_word_end = chunk[-1].offset + chunk[-1].duration
        remaining = chunk_end - last_word_end
        if remaining > 0.05:
            done_txt = (
                TextClip(
                    full_text,
                    fontsize=72,
                    color="white",
                    font="DejaVu-Sans-Bold",
                    method="caption",
                    size=(TARGET_W - 80, None),
                    stroke_color="black",
                    stroke_width=3,
                )
                .set_position(("center", 0.75), relative=True)
                .set_start(last_word_end)
                .set_duration(remaining)
            )
            layers.append(done_txt)

    return layers


# ── Video assembly (core) ──────────────────────────────────────────

def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_parts: List[Path],
    music_path: Optional[Path],
    word_timings: List[List[WordTiming]],
) -> Path:
    """Assemble final video: per-part clips + karaoke subtitles + voice + music."""
    if not clip_paths:
        raise RuntimeError("No video clips available. Check PEXELS_API_KEY / PIXABAY_API_KEY.")

    # 1. Load per-part audio, get durations
    part_audios = [AudioFileClip(str(p)) for p in audio_parts]
    durations = [a.duration for a in part_audios]
    total_duration = sum(durations)
    voice = concatenate_audioclips(part_audios)

    # 2. Distribute clips across parts
    if len(clip_paths) >= len(parts):
        chosen_clips = random.sample(clip_paths, len(parts))
    else:
        chosen_clips = clip_paths[:]
        random.shuffle(chosen_clips)
        while len(chosen_clips) < len(parts):
            chosen_clips.append(random.choice(clip_paths))

    # 3. Per-part: trim clip to audio duration, ken burns, karaoke subtitle
    source_clips = []
    video_clips = []
    for i, part in enumerate(parts):
        clip = VideoFileClip(str(chosen_clips[i]))
        source_clips.append(clip)
        dur = durations[i]

        fitted = _fit_clip_to_frame(clip, dur)
        fitted = _apply_ken_burns(fitted, dur)

        # Karaoke subtitles synced to word timing
        timings = word_timings[i] if i < len(word_timings) else []
        subtitle_layers = _make_karaoke_subtitle(timings, dur)

        composed = CompositeVideoClip(
            [fitted] + subtitle_layers,
            size=(TARGET_W, TARGET_H),
        ).set_duration(dur)
        video_clips.append(composed)

    # 4. Cross-fade between parts
    FADE_DUR = 0.2
    for idx in range(1, len(video_clips)):
        video_clips[idx] = video_clips[idx].crossfadein(FADE_DUR)

    video = concatenate_videoclips(video_clips, method="compose")
    video = video.set_duration(total_duration)

    # 5. Mix audio: voice + background music at 10% volume
    audio_tracks = [voice]
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.10)
        bg = bg.set_duration(total_duration)
        bg = bg.fx(afx.audio_fadeout, min(1.5, total_duration * 0.1))
        audio_tracks.append(bg)

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio).set_duration(total_duration)

    # 6. Render
    output_path = BUILD_DIR / "out.mp4"
    video.write_videofile(
        str(output_path),
        fps=30, codec="libx264", audio_codec="aac",
        preset="medium", bitrate="8000k", threads=4,
    )

    # Cleanup references
    for c in source_clips:
        try:
            c.close()
        except Exception:
            pass

    return output_path


# ── Pipeline ────────────────────────────────────────────────────────

def main() -> None:
    print("=== Omni Mystery Machine ===")
    _clean_build()

    print("[1/5] Generating script...")
    parts, metadata = call_groq_for_script()
    print(f"  Title: {metadata.title}")
    print(f"  Parts: {len(parts)}")

    print("[2/5] Downloading video clips...")
    clip_paths = download_pexels_clips()
    clip_paths += download_pixabay_clips()
    print(f"  Downloaded {len(clip_paths)} clips")

    print("[3/5] Generating TTS audio (per-part with word timings)...")
    audio_parts, word_timings = build_tts_per_part(parts)

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_parts, music_path, word_timings)
    print(f"  Video: {output}")

    upload_to_youtube(
        str(output),
        metadata.title,
        metadata.description,
        metadata.tags,
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()
