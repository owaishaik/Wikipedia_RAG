import streamlit as st
import requests
import wikipediaapi
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from st_audiorec import st_audiorec
from io import BytesIO
from llama_index.llms.cohere import Cohere

# Initialize Cohere LLM with API key
llm = Cohere(api_key="ntlBWBTl3UTkw45G4scwibmEr4iWgukgQ36ASGkO")

# Function to search for Wikipedia articles
def search_wikipedia(query):
    """Search for a Wikipedia page by title."""
    wiki_wiki = wikipediaapi.Wikipedia('generic-library/0.0')
    page = wiki_wiki.page(query)
    if page.exists():
        return page.fullurl, page.text
    return None, "No Wikipedia article found for the given query."

# Function to query transcription service
def transcribe_audio(audio_file):
    """Send the audio file to a transcription service and return the transcription."""
    response = requests.post('http://1.6.13.157:8122/transcribe', files={'file': ('audio.wav', audio_file, 'audio/wav')})
    if response.status_code == 200:
        return response.json().get('transcription', [''])[0]
    return None

# Function to query translation service
def translate_text(text, source_language="hin_Deva"):
    """Translate transcribed text to English."""
    response = requests.post('http://1.6.13.157:8122/translate', json={'text': text, 'source': source_language})
    if response.status_code == 200:
        return response.json().get('translation', '')
    return None

# Streamlit App Interface
st.title("Wikipedia Search & Live Audio Transcription App")

# Wikipedia Search Section
st.header("Search Wikipedia")
query = st.text_input("Enter a topic to search on Wikipedia")
if query:
    wikipedia_url, wikipedia_content = search_wikipedia(query)
    if wikipedia_url:
        st.write(f"Closest Wikipedia article: [Link]({wikipedia_url})")
        st.write(wikipedia_content[:500] + "...")  # Show a snippet of the content

        # Process the Wikipedia content with ChromaDB
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        doc = Document(text=wikipedia_content)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        # Initialize ChromaDB
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents([doc], storage_context=storage_context, embed_model=embed_model, transformations=[text_splitter])

        st.success("Vector Database Initialized Successfully!")

# Audio Transcription Section
st.header("Live Audio Transcription")
st.write("Record your audio below for transcription and processing.")

# Record audio
wav_audio_data = st_audiorec()

if wav_audio_data:
    st.success("Audio captured! Processing transcription...")

    # Convert audio data to a file-like object
    audio_file = BytesIO(wav_audio_data)
    audio_file.name = 'audio.wav'

    # Transcribe the audio
    transcription = transcribe_audio(audio_file)
    if transcription:
        st.write(f"Transcription: {transcription}")

        # Translate the transcription (if required)
        translation = translate_text(transcription)
        if translation:
            st.write(f"Translation: {translation}")

            # Query Vector DB using the transcription (if Vector DB is initialized)
            if 'index' in locals():
                query_engine = index.as_query_engine(llm=llm)
                response = query_engine.query(translation)  # Using translation for the query

                # Directly display the result of the query
                st.markdown(f"### Answer")
                st.write(response.response)
    else:
        st.error("Failed to transcribe the audio.")
