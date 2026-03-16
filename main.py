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
    visual_hint: str = ""  # one of VISUAL_CATEGORIES keys

@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: List[str]
    topic: str = ""


# ── Config ─────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_DIR = BUILD_DIR / "audio_parts"
MUSIC_PATH = BUILD_DIR / "music.mp3"
HISTORY_PATH = Path("topic_history.json")
MAX_HISTORY = 12  # remember last N topics to avoid repeats

TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-AndrewMultilingualNeural",
]
TTS_RATE_OPTIONS = ["+5%", "+8%", "+10%"]

# ── Hardcoded Pexels queries by category ───────────────────────────
PEXELS_QUERIES = [
    # Space
    "dark space galaxy nebula",
    "planet earth orbit night",
    "stars universe deep field",
    "astronaut space station dark",
    "moon surface crater closeup",
    "milky way timelapse night sky",
    "rocket launch space night",
    "satellite orbit earth dark",
    "solar system planets dark",
    "nebula colorful space deep",
    "mars red planet surface",
    "saturn rings planet space",
    "comet tail space dark",
    "jupiter clouds planet space",
    "telescope observatory night",
    "supernova star explosion space",
    "meteor shower night sky",
    "black hole visualization dark",
    "space shuttle launch fire",
    "earth sunrise from space",
    # Ocean / deep sea
    "ocean deep underwater dark",
    "underwater deep sea creatures",
    "coral reef ocean dark",
    "jellyfish underwater glowing",
    "whale underwater ocean deep",
    "submarine deep ocean dark",
    "underwater volcanic vent",
    "bioluminescent ocean creatures",
    # Mystery / ruins
    "ancient ruins mystery dark",
    "foggy forest mystery dark",
    "abandoned building dark mystery",
    "cave underground dark explore",
    "pyramid egypt ancient night",
    "dark tunnel underground mystery",
    "ancient temple jungle overgrown",
    "stone circle ancient monument",
    "hieroglyphs ancient wall carving",
    "old library dusty books dark",
    # Nature / extreme
    "aurora borealis northern lights",
    "volcano eruption night lava",
    "desert night sky stars",
    "ice glacier frozen landscape",
    "lightning storm dark sky",
    "tornado storm dark clouds",
    "waterfall jungle tropical mist",
    "arctic ice landscape frozen",
    # Technology / sci-fi
    "laboratory science experiment dark",
    "circuit board technology macro",
    "radar screen technology green",
    "data visualization abstract dark",
    "drone flying night lights",
    "robot arm technology dark",
    "hologram futuristic technology",
    "microscope science closeup",
    "dna molecule science animation",
]

# ── Visual category mapping (LLM picks a hint, we map to queries) ─
VISUAL_CATEGORIES = {
    "space": [
        "dark space galaxy nebula", "stars universe deep field",
        "nebula colorful space deep", "black hole visualization dark",
        "earth sunrise from space", "meteor shower night sky",
    ],
    "planet": [
        "planet earth orbit night", "mars red planet surface",
        "saturn rings planet space", "jupiter clouds planet space",
        "solar system planets dark", "moon surface crater closeup",
    ],
    "ocean": [
        "ocean deep underwater dark", "underwater deep sea creatures",
        "jellyfish underwater glowing", "bioluminescent ocean creatures",
        "submarine deep ocean dark", "underwater volcanic vent",
    ],
    "ruins": [
        "ancient ruins mystery dark", "pyramid egypt ancient night",
        "ancient temple jungle overgrown", "stone circle ancient monument",
        "hieroglyphs ancient wall carving", "old library dusty books dark",
    ],
    "mystery": [
        "foggy forest mystery dark", "abandoned building dark mystery",
        "cave underground dark explore", "dark tunnel underground mystery",
    ],
    "nature": [
        "aurora borealis northern lights", "volcano eruption night lava",
        "lightning storm dark sky", "tornado storm dark clouds",
        "ice glacier frozen landscape", "desert night sky stars",
    ],
    "technology": [
        "radar screen technology green", "data visualization abstract dark",
        "circuit board technology macro", "laboratory science experiment dark",
        "microscope science closeup", "dna molecule science animation",
    ],
    "astronaut": [
        "astronaut space station dark", "space shuttle launch fire",
        "rocket launch space night", "satellite orbit earth dark",
        "telescope observatory night",
    ],
}

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
    "aurora borealis sky",
    "submarine deep sea",
    "pyramid desert ancient",
    "meteor falling sky",
    "black hole space",
    "coral reef underwater",
    "cave dark explore",
    "rocket launch fire",
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

# ── Banned filler phrases for quality gate ─────────────────────────
_FILLER_PATTERNS = [
    "you won't believe",
    "this is amazing",
    "trust me",
    "here's the thing",
    "let that sink in",
    "think about that",
    "are you ready",
    "brace yourself",
    "in this video",
    "stay tuned",
    "subscribe for more",
    "like and share",
]

# ── Mystery/space topics for prompt variety (100+) ────────────────
TOPICS = [
    # Space & cosmos
    "black holes and singularities",
    "dark matter and dark energy",
    "the Fermi Paradox",
    "rogue planets wandering the galaxy",
    "the Great Attractor pulling galaxies",
    "quantum entanglement and teleportation",
    "the multiverse theory",
    "the Dyson Sphere concept",
    "the Voyager golden record",
    "magnetars and neutron stars",
    "the simulation hypothesis",
    "gamma ray bursts destroying galaxies",
    "the dark side of the Moon",
    "the Bootes Void — emptiest place in space",
    "Oumuamua — the interstellar visitor",
    "Tabby's Star and the alien megastructure theory",
    "the Great Silence — why can't we hear aliens",
    "the Kessler Syndrome — space junk apocalypse",
    "the heat death of the universe",
    "time dilation near black holes",
    "the Kardashev scale — levels of civilization",
    "the Pale Blue Dot photo by Voyager 1",
    "white holes — the opposite of black holes",
    "the Big Crunch vs Big Rip theories",
    "Alcubierre warp drive — FTL travel",
    "the Hubble Deep Field image",
    "the cosmic microwave background radiation",
    "Sagittarius A* — our galaxy's supermassive black hole",
    "the Oort Cloud at the edge of our solar system",
    "Enceladus and its underground ocean",
    "Titan — the only moon with an atmosphere",
    "Europa and potential alien life under ice",
    "the Great Red Spot of Jupiter",
    "solar flares and Carrington events",
    "the asteroid that killed the dinosaurs",
    "Planet Nine — the hidden giant",
    "pulsars and their lighthouse beams",
    "the heliopause — where our Sun's influence ends",
    "dark flow — galaxies moving toward something invisible",
    "the Observable Universe — 46 billion light years",
    # Ocean & deep sea
    "deep ocean unexplored zones",
    "the Mariana Trench deepest point",
    "bioluminescent life in the deep ocean",
    "the Bloop — mysterious ocean sound",
    "giant squid — real sea monsters",
    "deep sea hydrothermal vents and extremophiles",
    "the Mid-Atlantic Ridge underwater mountains",
    "underwater rivers and waterfalls in the ocean",
    "ocean dead zones where nothing survives",
    "the Bermuda Triangle",
    "the Baltic Sea anomaly",
    "phantom islands that appeared on maps but don't exist",
    # Mystery & unexplained
    "Area 51 and UFO sightings",
    "the Wow signal from space",
    "the Tunguska event 1908",
    "the Nazca Lines of Peru",
    "the Antikythera mechanism",
    "the Eye of the Sahara / Richat Structure",
    "ancient lost civilizations",
    "uncontacted tribes on Earth",
    "the Voynich Manuscript nobody can decode",
    "the Dyatlov Pass incident",
    "the Zodiac Killer's unsolved ciphers",
    "the Mary Celeste ghost ship",
    "Stonehenge and how it was built",
    "the Piri Reis map showing Antarctica before discovery",
    "the Chelyabinsk meteor 2013",
    "the Hessdalen lights in Norway",
    "the Taos Hum — a sound only some people hear",
    "the Oak Island Money Pit",
    "the disappearance of the Roanoke colony",
    "DB Cooper — the only unsolved hijacking",
    "the Somerton Man and Tamam Shud case",
    "Gobekli Tepe — 12,000-year-old temple complex",
    "the Baghdad Battery — ancient electricity",
    "the Copper Scroll treasure map",
    "the lost city of Atlantis theories",
    "the Bermeja island that vanished from maps",
    # Science & physics
    "the double slit experiment and consciousness",
    "the Mandela Effect and false memories",
    "CRISPR gene editing and designer humans",
    "the speed of light as a universal limit",
    "antimatter and why it's the most expensive substance",
    "the Higgs Boson and the God Particle",
    "string theory and extra dimensions",
    "the Grandfather Paradox of time travel",
    "quantum tunneling through solid walls",
    "the Mpemba effect — hot water freezing faster",
    "Boltzmann brains floating in empty space",
    "the Vacuum Catastrophe of quantum physics",
    # Biology & Earth
    "tardigrades — the indestructible animals",
    "the Chicxulub crater and mass extinction",
    "supervolcano Yellowstone and when it erupts",
    "the Permian extinction — 96% of life died",
    "the Tsar Bomba — most powerful explosion ever",
    "the Doomsday Vault in Svalbard Norway",
    "Lake Vostok buried under 2 miles of Antarctic ice",
    "the Sentinelese — most isolated people on Earth",
    "extremophile bacteria living inside nuclear reactors",
    "the Sahara Desert was once a green jungle",
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
    "what happens when you go too deep",
    "the biggest lie they told you about space",
    "the discovery that changed everything",
    "why this terrifies even NASA scientists",
    "the hidden truth nobody talks about",
    "the experiment that went horribly wrong",
    "the place on Earth no human can survive",
    "the signal that came from nowhere",
]


# ── Helpers ─────────────────────────────────────────────────────────

def _clean_build() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _load_topic_history() -> list:
    if HISTORY_PATH.is_file():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_topic_history(history: list) -> None:
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def _pick_unique_topic() -> str:
    """Pick a topic not used in last MAX_HISTORY runs, weighted by performance."""
    from analytics import get_topic_weights

    history = _load_topic_history()
    available = [t for t in TOPICS if t not in history]
    if not available:
        history = []
        available = list(TOPICS)
    weights = get_topic_weights(available)
    if weights:
        topic = random.choices(available, weights=weights, k=1)[0]
    else:
        topic = random.choice(available)
    history.append(topic)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    _save_topic_history(history)
    return topic


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


def _validate_script(parts: List[ScriptPart]) -> bool:
    """Quality gate — rejects weak/generic scripts."""
    if len(parts) < 8:
        print(f"[QUALITY] Rejected: too few parts ({len(parts)}, need >=8)")
        return False

    avg_words = sum(len(p.text.split()) for p in parts) / len(parts)
    if avg_words < 8:
        print(f"[QUALITY] Rejected: avg words too low ({avg_words:.1f}, need >=8)")
        return False

    filler_count = 0
    for part in parts:
        text_lower = part.text.lower()
        for filler in _FILLER_PATTERNS:
            if filler in text_lower:
                filler_count += 1
                break
    if filler_count > 2:
        print(f"[QUALITY] Rejected: too many fillers ({filler_count})")
        return False

    # At least 40% of phrases must contain specific content
    concrete = re.compile(
        r'\d|light.?year|billion|million|trillion|kilometer|mile|'
        r'degree|megahertz|frequency|nasa|voyager|hubble|'
        r'ocean|trench|depth|signal|radiation|gravity|orbit|'
        r'century|ancient|civilization|species|temperature|'
        r'scientist|researcher|telescope|satellite|galaxy|'
        r'neutron|magnetic|solar|lunar|crater|extinction',
        re.IGNORECASE,
    )
    concrete_count = sum(1 for p in parts if concrete.search(p.text))
    ratio = concrete_count / len(parts)
    if ratio < 0.4:
        print(f"[QUALITY] Rejected: not enough concrete content ({ratio:.0%}, need >=40%)")
        return False

    print(f"[QUALITY] Passed: {len(parts)} parts, avg {avg_words:.1f} words, {ratio:.0%} concrete")
    return True


# ── LLM — multi-part script ────────────────────────────────────────

def call_groq_for_script() -> tuple:
    """Generate a multi-part mystery/space script via Groq. Returns (parts, metadata)."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    topic = _pick_unique_topic()
    angle = random.choice(ANGLES)
    print(f"  Topic: {topic} | Angle: {angle}")

    system_prompt = (
        "You are an expert scriptwriter for a viral YouTube Shorts channel called "
        "'Void Chronicles AI'. You create TERRIFYING, mind-blowing scripts about space, "
        "deep ocean, unsolved mysteries, and conspiracy theories. "
        "Every phrase must deliver a SPECIFIC fact: real names, dates, numbers, distances. "
        "NEVER write filler like 'This is amazing' or 'You won't believe this'. "
        "Use a dark, cinematic narration tone — like a documentary narrator revealing secrets. "
        "Respond ONLY with valid JSON, no markdown wrappers."
    )

    _valid_cats = ", ".join(sorted(VISUAL_CATEGORIES.keys()))

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
8. For each part, pick a "visual_hint" — the background video category that fits the phrase best.
   Valid categories: {_valid_cats}

Format — strictly JSON:
{{
  "title": "Catchy YouTube title with emoji (max 70 chars) #Shorts",
  "description": "YouTube description (2–3 lines) with hashtags",
  "tags": ["space", "mystery", "shorts", ...4-7 more specific tags],
  "parts": [
    {{ "text": "Phrase with specific terrifying fact, 12-25 words", "visual_hint": "space" }}
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

            parts = [
                ScriptPart(
                    text=p["text"],
                    visual_hint=p.get("visual_hint", "space"),
                )
                for p in data.get("parts", []) if p.get("text")
            ]
            metadata = VideoMetadata(
                title=data.get("title", "")[:100] or "Mystery of the Universe #Shorts",
                description=data.get("description", "") or "#shorts #mystery #space",
                tags=data.get("tags", ["space", "mystery", "shorts"]),
                topic=topic,
            )
            metadata = _enrich_metadata(metadata)

            if _validate_script(parts):
                return parts, metadata
            print(f"[WARN] Script failed quality check (attempt {attempt + 1})")

        except Exception as exc:
            print(f"[WARN] Groq error (attempt {attempt + 1}): {exc}")

        body["temperature"] = 1.0

    # ── Retry with reinforced prompt ──
    body["messages"].append({
        "role": "user",
        "content": (
            "IMPORTANT: the previous response failed quality checks. "
            "Make sure:\n"
            "1. At least 10 parts, each 12-25 words.\n"
            "2. Every part has SPECIFIC content: real names, dates, numbers, distances, measurements.\n"
            "3. NO filler phrases like 'You won't believe' or 'This is amazing'.\n"
            "Return JSON in the same format."
        ),
    })
    try:
        chat2 = client.chat.completions.create(**body)
        raw2 = chat2.choices[0].message.content
        raw2 = re.sub(r"^```(?:json)?\s*", "", raw2.strip())
        raw2 = re.sub(r"\s*```$", "", raw2.strip())
        data2 = json.loads(raw2)
        parts2 = [
            ScriptPart(
                text=p["text"],
                visual_hint=p.get("visual_hint", "space"),
            )
            for p in data2.get("parts", []) if p.get("text")
        ]
        metadata2 = VideoMetadata(
            title=data2.get("title", "")[:100] or "Mystery of the Universe #Shorts",
            description=data2.get("description", "") or "#shorts #mystery #space",
            tags=data2.get("tags", ["space", "mystery", "shorts"]),
            topic=topic,
        )
        metadata2 = _enrich_metadata(metadata2)
        if _validate_script(parts2):
            return parts2, metadata2
        print("[WARN] Reinforced retry also failed, using fallback")
    except Exception as exc:
        print(f"[WARN] Reinforced retry error: {exc}, using fallback")

    return _fallback_script()


def _fallback_script() -> tuple:
    """4 hardcoded scripts for variety when LLM fails."""
    _POOL = [
        # 1 — The Wow Signal
        (
            [
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
            ],
            VideoMetadata(
                title="📡 The WOW Signal Still Can't Be Explained #Shorts",
                description="In 1977, we received a message from space that lasted 72 seconds. It has never been explained. 📡",
                tags=["space", "mystery", "wow signal", "aliens", "shorts", "science"],
                topic="the Wow Signal from 1977",
            ),
        ),
        # 2 — The Bootes Void
        (
            [
                ScriptPart("There is a region of space 330 million light years across that contains almost nothing."),
                ScriptPart("It's called the Bootes Void, and it should contain about 2,000 galaxies. It has only 60."),
                ScriptPart("That's 97 percent emptier than any other region of comparable size in the observable universe."),
                ScriptPart("If our Milky Way were at the center of the Bootes Void, we wouldn't have known other galaxies existed until the 1960s."),
                ScriptPart("Scientists have no accepted explanation for why this void is so impossibly empty."),
                ScriptPart("Some physicists suggest it could be evidence of a Type III civilization that consumed all available matter."),
                ScriptPart("Others believe it formed from the merging of smaller voids over 10 billion years."),
                ScriptPart("The void is expanding faster than the surrounding universe, and nobody knows why."),
                ScriptPart("If you traveled at the speed of light, it would take 330 million years just to cross it."),
                ScriptPart("The Bootes Void is the loneliest place in the known universe, and it's getting lonelier."),
            ],
            VideoMetadata(
                title="🕳️ The Emptiest Place in Space Is Terrifying #Shorts",
                description="The Bootes Void is 330 million light years of almost nothing. Scientists can't explain it. 🕳️",
                tags=["space", "bootes void", "mystery", "cosmos", "shorts", "science"],
                topic="the Bootes Void",
            ),
        ),
        # 3 — The Mariana Trench
        (
            [
                ScriptPart("The deepest point on Earth is 36,000 feet below the ocean surface. It's called Challenger Deep."),
                ScriptPart("The water pressure there is 1,000 times what you feel on the surface. It would crush a human instantly."),
                ScriptPart("Only three manned expeditions have ever reached the bottom. We've sent more people to the Moon."),
                ScriptPart("In 2019, explorer Victor Vescovo found a plastic bag at the bottom. Pollution reaches everywhere."),
                ScriptPart("Scientists discovered life forms at the bottom that survive without sunlight, feeding on chemicals from the Earth's crust."),
                ScriptPart("Temperatures near hydrothermal vents exceed 700 degrees Fahrenheit, yet organisms thrive there."),
                ScriptPart("Over 80 percent of the ocean floor remains completely unmapped and unexplored."),
                ScriptPart("Sonar readings have detected massive unknown shapes moving in the deep that don't match any known species."),
                ScriptPart("The Mariana Trench is growing wider by about 2.5 centimeters every year as tectonic plates shift."),
                ScriptPart("We know more about the surface of Mars than we do about our own ocean floor."),
            ],
            VideoMetadata(
                title="🌊 36,000 Feet Deep — What Lives Down There? #Shorts",
                description="The Mariana Trench is the deepest place on Earth. What we found there is terrifying. 🌊",
                tags=["ocean", "mariana trench", "deep sea", "mystery", "shorts", "science"],
                topic="the Mariana Trench",
            ),
        ),
        # 4 — Oumuamua
        (
            [
                ScriptPart("In October 2017, astronomers detected the first known interstellar object passing through our solar system."),
                ScriptPart("They named it Oumuamua, meaning 'scout' in Hawaiian. It was shaped like nothing ever seen before."),
                ScriptPart("It was 10 times longer than it was wide — like a 400-meter cigar tumbling through space."),
                ScriptPart("Oumuamua accelerated as it left our solar system, which cannot be explained by gravity alone."),
                ScriptPart("Harvard astronomer Avi Loeb proposed it could be an alien solar sail or probe."),
                ScriptPart("The object didn't produce a visible comet tail, ruling out standard outgassing explanations."),
                ScriptPart("It entered our solar system from the direction of the star Vega, traveling at 196,000 miles per hour."),
                ScriptPart("By the time we detected it, Oumuamua was already heading away. We had weeks to study it."),
                ScriptPart("No telescope on Earth or in space has been able to determine what it was made of."),
                ScriptPart("Oumuamua is now beyond our reach, speeding into interstellar space. We may never know what it truly was."),
            ],
            VideoMetadata(
                title="🛸 The Alien Object That Flew Through Our Solar System #Shorts",
                description="In 2017, something entered our solar system from interstellar space. Scientists still can't explain it. 🛸",
                tags=["oumuamua", "aliens", "space", "interstellar", "mystery", "shorts", "science"],
                topic="Oumuamua interstellar object",
            ),
        ),
    ]
    idx = random.randrange(len(_POOL))
    parts, meta = _POOL[idx]
    meta = _enrich_metadata(meta)
    print(f"[FALLBACK] Using fallback script #{idx + 1}")
    return parts, meta


_DESCRIPTION_FOOTER = (
    "\n\n#shorts #mystery #space #science #facts #universe"
    "\nSubscribe to Void Chronicles AI for daily mind-blowing facts!"
)


def _enrich_metadata(meta: VideoMetadata) -> VideoMetadata:
    """Ensure SEO essentials are present."""
    if "#shorts" not in meta.title.lower():
        meta.title = meta.title[:90] + " #Shorts"
    core_tags = {"shorts", "mystery", "space", "science", "facts"}
    existing = {t.lower().strip() for t in meta.tags}
    for tag in core_tags - existing:
        meta.tags.append(tag)
    if "#mystery" not in meta.description.lower():
        meta.description += _DESCRIPTION_FOOTER
    return meta


# ── Clip downloads ──────────────────────────────────────────────────

def download_pexels_clips(
    parts: List[ScriptPart], target_count: int = 14,
) -> dict:
    """Download clips from Pexels matched to visual categories.

    Returns dict mapping category -> list of clip Paths.
    Also includes a "_pool" key with all clips for fallback.
    """
    if not PEXELS_API_KEY:
        return {"_pool": []}

    headers = {"Authorization": PEXELS_API_KEY}

    # Figure out which categories we need clips for
    needed_cats: dict = {}
    for part in parts:
        cat = part.visual_hint if part.visual_hint in VISUAL_CATEGORIES else "space"
        needed_cats[cat] = needed_cats.get(cat, 0) + 1

    # Build query list: prioritize category queries, then fill with general pool
    query_plan: list = []  # (query, category)
    for cat, count in needed_cats.items():
        cat_queries = list(VISUAL_CATEGORIES.get(cat, PEXELS_QUERIES[:5]))
        random.shuffle(cat_queries)
        for q in cat_queries[:count]:
            query_plan.append((q, cat))

    # Fill remainder from general pool
    used_queries = {q for q, _ in query_plan}
    general = [q for q in PEXELS_QUERIES if q not in used_queries]
    random.shuffle(general)
    for q in general:
        if len(query_plan) >= target_count:
            break
        query_plan.append((q, "_general"))

    result: dict = {"_pool": []}
    seen_ids: set = set()
    clip_idx = 0

    for query, cat in query_plan:
        if clip_idx >= target_count:
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
                result.setdefault(cat, []).append(clip_path)
                result["_pool"].append(clip_path)
                print(f"    Pexels [{query}] ({cat}) -> clip {clip_idx}")
            except Exception as exc:
                print(f"[WARN] Pexels clip {clip_idx} download failed: {exc}")
            break

    return result


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
    word_timings: List[WordTiming], duration: float, is_hook: bool = False,
) -> list:
    """Karaoke-style subtitles: groups of 2-3 words appear in sync with speech.

    Each group appears when the first word is spoken and stays until the last
    word finishes. Current group is bright YELLOW, then fades to WHITE before
    the next group appears. Creates the classic "word pop" effect.

    If is_hook=True, uses larger fontsize and red highlight for the first part.
    """
    if not word_timings:
        return []

    # Group words into chunks of 2-3 for "pop" effect
    CHUNK_SIZE = 3
    chunks = []
    for i in range(0, len(word_timings), CHUNK_SIZE):
        chunks.append(word_timings[i:i + CHUNK_SIZE])

    layers = []
    for ci, chunk in enumerate(chunks):
        chunk_start = chunk[0].offset
        # End = next chunk start, or end of last word + 0.2s
        if ci + 1 < len(chunks):
            chunk_end = chunks[ci + 1][0].offset
        else:
            chunk_end = min(chunk[-1].offset + chunk[-1].duration + 0.3, duration)
        chunk_dur = chunk_end - chunk_start
        if chunk_dur <= 0:
            continue

        full_text = " ".join(w.text for w in chunk)

        # Hook styling: larger font + red for first part
        active_color = "#FF4444" if is_hook else "yellow"
        active_fontsize = 88 if is_hook else 72
        fade_fontsize = 80 if is_hook else 72

        # Active highlight while words are being spoken
        speak_end = chunk[-1].offset + chunk[-1].duration
        speak_dur = speak_end - chunk_start
        if speak_dur > 0:
            try:
                yellow_txt = (
                    TextClip(
                        full_text,
                        fontsize=active_fontsize,
                        color=active_color,
                        font="DejaVu-Sans-Bold",
                        method="caption",
                        size=(TARGET_W - 100, None),
                        stroke_color="black",
                        stroke_width=4,
                    )
                    .set_position(("center", 0.75), relative=True)
                    .set_start(chunk_start)
                    .set_duration(min(speak_dur, chunk_dur))
                )
                layers.append(yellow_txt)
            except Exception as exc:
                print(f"[WARN] Karaoke TextClip failed: {exc}")

        # White after spoken (brief pause before next group)
        remaining = chunk_end - speak_end
        if remaining > 0.05:
            try:
                white_txt = (
                    TextClip(
                        full_text,
                        fontsize=fade_fontsize,
                        color="white",
                        font="DejaVu-Sans-Bold",
                        method="caption",
                        size=(TARGET_W - 100, None),
                        stroke_color="black",
                        stroke_width=3,
                    )
                    .set_position(("center", 0.75), relative=True)
                    .set_start(speak_end)
                    .set_duration(remaining)
                )
                layers.append(white_txt)
            except Exception:
                pass

    return layers


# ── Video assembly (core) ──────────────────────────────────────────

def build_video(
    parts: List[ScriptPart],
    clip_map: dict,
    pixabay_clips: List[Path],
    audio_parts: List[Path],
    music_path: Optional[Path],
    word_timings: List[List[WordTiming]],
) -> Path:
    """Assemble final video: per-part clips + karaoke subtitles + voice + music."""
    all_clips = clip_map.get("_pool", []) + pixabay_clips
    if not all_clips:
        raise RuntimeError("No video clips available. Check PEXELS_API_KEY / PIXABAY_API_KEY.")

    # 1. Load per-part audio, get durations
    part_audios = [AudioFileClip(str(p)) for p in audio_parts]
    durations = [a.duration for a in part_audios]
    total_duration = sum(durations)
    voice = concatenate_audioclips(part_audios)

    # 2. Assign clips per part — prefer matching visual category
    chosen_clips: List[Path] = []
    used_clips: set = set()
    for part in parts:
        cat = part.visual_hint if part.visual_hint in VISUAL_CATEGORIES else "space"
        candidates = [c for c in clip_map.get(cat, []) if c not in used_clips]
        if not candidates:
            candidates = [c for c in all_clips if c not in used_clips]
        if not candidates:
            candidates = all_clips  # reuse if exhausted
        pick = random.choice(candidates)
        chosen_clips.append(pick)
        used_clips.add(pick)

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
        subtitle_layers = _make_karaoke_subtitle(timings, dur, is_hook=(i == 0))
        print(f"    Part {i}: {len(timings)} word timings → {len(subtitle_layers)} subtitle layers")

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
    clip_map = download_pexels_clips(parts)
    pixabay_clips = download_pixabay_clips()
    total_clips = len(clip_map.get("_pool", [])) + len(pixabay_clips)
    print(f"  Downloaded {total_clips} clips")

    print("[3/5] Generating TTS audio (per-part with word timings)...")
    audio_parts, word_timings = build_tts_per_part(parts)

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_map, pixabay_clips, audio_parts, music_path, word_timings)
    print(f"  Video: {output}")

    video_id = upload_to_youtube(
        str(output),
        metadata.title,
        metadata.description,
        metadata.tags,
    )
    from analytics import log_upload
    log_upload(video_id, metadata.title, metadata.topic, metadata.tags)
    print("=== Done ===")


if __name__ == "__main__":
    main()
