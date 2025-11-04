from langchain_community.document_loaders import YoutubeLoader 
import os

def Youtube_loader_transcript(video_url:str , save_path: str = "data/transcript"):
    os.makedirs(save_path,exist_ok=True)
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info = True,
        language="en",
    )
    document = loader.load()

    file_path = os.path.join(save_path, f"{video_url.split('=')[-1]}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in document:
            f.write(doc.page_content)
    return document

