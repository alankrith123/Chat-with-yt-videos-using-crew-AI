import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from transformers import pipeline

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="youtube_content")

nlp = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def search_youtube(query, max_results=20):
    try:
        request = youtube.search().list(
            q=query,
            type='video',
            part='id,snippet',
            maxResults=max_results,
            order='date'
        )
        response = request.execute()
        return [{'id': item['id']['videoId'], 'title': item['snippet']['title']} for item in response.get('items', [])]
    except Exception:
        return []

def scrape_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry["text"] for entry in transcript])
    except Exception:
        return ""

def add_to_rag(video_id, title, content):
    collection.add(
        documents=[content],
        metadatas=[{"video_id": video_id, "title": title}],
        ids=[video_id]
    )

def chat_with_rag(query):
    results = collection.query(query_texts=[query], n_results=3)
    if not results or 'documents' not in results or not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant content found in the database. Try searching for another topic."
    context = ' '.join(results['documents'][0])
    prompt = f"Using the following video content, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = nlp(
            prompt,
            max_length=400,
            min_length=150,
            do_sample=True,
            temperature=0.8
        )[0]['generated_text']
        return response
    except Exception:
        return "The question you asked is not relevant to the stored content."

researcher = Agent(
    role="Researcher",
    goal="Find relevant YouTube videos on a given topic",
    backstory="You are an expert at finding relevant and recent information on YouTube.",
    tools=[Tool(name="YouTube Search", func=search_youtube, description="Search YouTube for recent videos on a topic.")]
)

scraper = Agent(
    role="Scraper",
    goal="Extract content from YouTube videos and add to RAG",
    backstory="You are skilled at extracting and processing video transcripts.",
    tools=[
        Tool(name="YouTube Transcript Scraper", func=scrape_youtube_transcript, description="Extract transcript from a YouTube video."),
        Tool(name="RAG Tool", func=add_to_rag, description="Store extracted YouTube content in a RAG database.")
    ]
)

chat_agent = Agent(
    role="Chat Agent",
    goal="Answer questions based on the scraped YouTube content",
    backstory="You are an AI assistant that provides information based on YouTube video content.",
    tools=[Tool(name="RAG Chat", func=chat_with_rag, description="Answer questions based on stored YouTube content.")]
)

search_task = Task(
    description="Search for the 20 most recent YouTube videos on a specific topic.",
    agent=researcher,
    expected_output="A list of YouTube video titles and IDs related to the given topic."
)

scrape_task = Task(
    description="Scrape content from the found YouTube videos and add to RAG.",
    agent=scraper,
    expected_output="Extracted transcripts of YouTube videos stored in the RAG database."
)

chat_task = Task(
    description="Chat about the scraped content.",
    agent=chat_agent,
    expected_output="A response based on the stored YouTube video content."
)

youtube_crew = Crew(
    agents=[researcher, scraper, chat_agent],
    tasks=[search_task, scrape_task, chat_task],
    process=Process.sequential
)

def main():
    print("YouTube Content Scraper & Chat Bot")
    topic = input("Enter a topic to search for on YouTube: ")
    videos = youtube_crew.tasks[0].agent.tools[0].func(topic)
    if not videos:
        print("No videos found. Exiting...")
        return
    for video in videos:
        content = youtube_crew.tasks[1].agent.tools[0].func(video["id"])
        if content:
            youtube_crew.tasks[1].agent.tools[1].func(video["id"], video["title"], content)
        else:
            print(f"Skipping video: {video['title']} (No transcript)")
    print("Chat session started. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = youtube_crew.tasks[2].agent.tools[0].func(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
