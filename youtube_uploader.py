import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials


def upload_to_youtube(file_path, title, description, tags):
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("YT_CLIENT_ID"),
        client_secret=os.getenv("YT_CLIENT_SECRET"),
    )

    youtube = build("youtube", "v3", credentials=creds)

    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags.split(","),
            "categoryId": "22",
        },
        "status": {"privacyStatus": "public"},
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    response = youtube.videos().insert(
        part="snippet,status", body=request_body, media_body=media
    ).execute()
    video_id = response.get("id", "")
    print(f"Uploaded! https://youtube.com/shorts/{video_id}")
