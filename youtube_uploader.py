import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials


def upload_to_youtube(file_path, title, description, tags):
    print(f"Uploading: {title}")

    creds = Credentials(
        token=None,
        refresh_token=os.getenv("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("YT_CLIENT_ID"),
        client_secret=os.getenv("YT_CLIENT_SECRET"),
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

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )

    response = request.execute()
    video_id = response["id"]
    print(f"Uploaded! https://youtube.com/shorts/{video_id}")
    return video_id
