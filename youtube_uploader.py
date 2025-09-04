import os, json, time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

YT_CLIENT_ID = os.environ["YT_CLIENT_ID"]
YT_CLIENT_SECRET = os.environ["YT_CLIENT_SECRET"]
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"]

TOKEN_URI = "https://oauth2.googleapis.com/token"

def yt_client():
    creds = Credentials(
        token=None,
        refresh_token=YT_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=YT_CLIENT_ID,
        client_secret=YT_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    return build("youtube","v3", credentials=creds)

def upload_video(file_path, title, description, tags, category_id="19", privacy_status="public", thumbnail_path=None):
    youtube = yt_client()
    body = {
        "snippet": {
            "title": title[:95],
            "description": description[:4800],
            "tags": tags[:30],
            "categoryId": category_id
        },
        "status": {"privacyStatus": privacy_status}
    }
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status: print(f"Upload {int(status.progress()*100)}%")
    video_id = response["id"]
    print("Video uploaded:", video_id)
    if thumbnail_path:
        youtube.thumbnails().set(videoId=video_id, media_body=thumbnail_path).execute()
        print("Thumbnail set.")
    return video_id
