import os
import time

import requests as _requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

TOKEN_URL = "https://oauth2.googleapis.com/token"


def _verify_refresh_token(client_id: str, client_secret: str, refresh_token: str):
    """Pre-check: exchange refresh token for access token to detect invalid_grant early."""
    resp = _requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }, timeout=30)
    data = resp.json()
    if "error" in data:
        err = data.get("error", "")
        desc = data.get("error_description", "")
        if err == "invalid_grant":
            raise RuntimeError(
                f"OAuth invalid_grant: {desc}. "
                "The refresh token has expired or been revoked. "
                "Re-run: python get_refresh_token.py  then update the "
                "YT_REFRESH_TOKEN secret in GitHub repo settings."
            )
        raise RuntimeError(f"OAuth error: {err} — {desc}")


def upload_to_youtube(file_path, title, description, tags):
    print(f"Uploading: {title}")

    client_id = os.getenv("YT_CLIENT_ID")
    client_secret = os.getenv("YT_CLIENT_SECRET")
    refresh_token = os.getenv("YT_REFRESH_TOKEN")

    _verify_refresh_token(client_id, client_secret, refresh_token)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URL,
        client_id=client_id,
        client_secret=client_secret,
    )

    youtube = build("youtube", "v3", credentials=creds)

    # Handle tags as either comma-separated string or list
    if isinstance(tags, str):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, list):
        tag_list = tags
    else:
        tag_list = ["shorts", "mystery", "space"]

    privacy = os.getenv("YOUTUBE_PRIVACY", "public")
    if privacy not in ("public", "unlisted", "private"):
        privacy = "public"

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tag_list,
            "categoryId": "28",  # Science & Technology
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }

    for attempt in range(1, 4):
        try:
            media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(
                part="snippet,status", body=body, media_body=media
            )
            response = request.execute()
            video_id = response["id"]
            print(f"Uploaded! https://youtube.com/shorts/{video_id}")
            return video_id
        except HttpError as exc:
            if exc.resp.status < 500:
                raise
            print(f"[WARN] Upload attempt {attempt}/3: HTTP {exc.resp.status}")
            if attempt < 3:
                time.sleep(attempt * 15)
        except Exception as exc:
            print(f"[WARN] Upload attempt {attempt}/3: {exc}")
            if attempt < 3:
                time.sleep(attempt * 15)
    raise RuntimeError("Upload failed after 3 attempts")
