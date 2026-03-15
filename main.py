import os
import asyncio
import json
import requests
import random
from groq import Groq
import pexels_api
import edge_tts
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, vfx,
    AudioFileClip, TextClip, CompositeVideoClip,
)
from youtube_uploader import upload_to_youtube

from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")


def get_ai_content():
    client = Groq(api_key=GROQ_API_KEY)
    prompt = """
    Generate a mind-blowing space or mystery fact for YouTube Shorts.
    Return ONLY a JSON object:
    {
      "fact": "Text of the fact (max 15 words)",
      "search_term": "one specific keyword for luxury/space/mystery video",
      "seo": {
        "title": "Viral title with emojis",
        "desc": "Description with #shorts #ai #mystery",
        "tags": "ai, mystery, space, facts"
      }
    }
    """
    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        response_format={"type": "json_object"},
    )
    return json.loads(chat.choices[0].message.content)


def download_video_assets(query):
    api = pexels_api.API(PEXELS_API_KEY)
    search = api.search_videos(query=query, orientation="portrait", per_page=5)
    video_path = "source_video.mp4"
    if search["videos"]:
        url = search["videos"][0]["video_files"][0]["link"]
        with open(video_path, "wb") as f:
            f.write(requests.get(url, timeout=120).content)
    return video_path


async def generate_voice(text):
    path = "voice.mp3"
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(path)
    return path


def create_final_video(video_p, audio_p, text):
    video = VideoFileClip(video_p).resize(height=1920)
    audio = AudioFileClip(audio_p)

    duration = audio.duration
    final_v = video.subclip(0, min(video.duration, duration)).fx(vfx.speedx, 0.9)

    if final_v.duration < duration:
        final_v = final_v.loop(duration=duration)
    else:
        final_v = final_v.set_duration(duration)

    txt = (
        TextClip(
            text,
            fontsize=70,
            color="yellow",
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(800, None),
        )
        .set_duration(duration)
        .set_pos("center")
    )

    final_clip = CompositeVideoClip([final_v, txt]).set_audio(audio)
    output = "final_shorts.mp4"
    final_clip.write_videofile(output, fps=24, codec="libx264", audio_codec="aac")
    return output


async def run_process():
    print("--- Start Engine ---")
    content = get_ai_content()
    print(f"  Fact: {content['fact']}")
    print(f"  Search: {content['search_term']}")

    v_file = download_video_assets(content["search_term"])
    a_file = await generate_voice(content["fact"])
    final_mp4 = create_final_video(v_file, a_file, content["fact"])

    upload_to_youtube(
        final_mp4,
        content["seo"]["title"],
        content["seo"]["desc"],
        content["seo"]["tags"],
    )
    print("--- Done ---")


if __name__ == "__main__":
    asyncio.run(run_process())
