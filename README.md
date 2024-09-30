### Wikipedia Search & Live Audio Transcription App

To scrape the text of a Wikipedia article, I used the Wikipedia API through the wikipediaapi Python library. This library allows fetching the full text of a Wikipedia article by querying it with the title. It also provides easy access to various parts of the article, like the summary, sections, and links.

I used ChromaDB to store and query vector embeddings. After processing the Wikipedia content into smaller chunks and generating embeddings using the HuggingFace model, I store these vectors in ChromaDB to allow for fast and efficient similarity searches. This makes it easier to find relevant information based on the transcribed and translated audio queries.

## Setup Instructions

First, create and activate a virtual environment to isolate the dependencies of the project.

```
python -m venv env

```

```
.\env\Scripts\activate

```

Then, install all the required dependencies listed in the requirements.txt file

```
pip install -r requirements.txt

```

Then, you can start the app by running the following command

```
streamlit run st.py

```

## Server Setup for Transcription and Translation

Since my local machine lacks a GPU and is not Linux-based, I run AI4Bharat’s IndicConformer (for transcription) and IndicTrans2 (for translation) models on my A40 GPU-powered Linux server. The models are served on the server, and I query them via HTTP requests.

To connect to the server and keep the models running, I use the following command in my server terminal:

```

screen -r 1220649.pts-12.aryabhata

```

#### NOTE
When running the application on your local machine, change the server IP (1.6.13.157) to localhost in both the transcribe_audio and translate_text functions (in st.py file) if the services are running locally. For example:

```

response = requests.post('http://localhost:8122/transcribe', files={'file': ('audio.wav', audio_file, 'audio/wav')})

```

## Major Challenges Faced

One of the primary hurdles was the absence of a NVIDIA GPU on my local machine. My laptop is quite old and lacks the hardware required to efficiently run GPU-dependent models like IndicConformer and IndicTrans2, which are crucial for transcription and translation in this project. Without a GPU, tasks like processing large audio files or running machine learning models can be extremely slow or infeasible on a typical CPU.

To overcome this limitation, I leveraged a remote A40 GPU-powered server to handle the heavy computational workload. This setup allowed me to offload the transcription and translation tasks to the server while interacting with it through HTTP requests. Though this workaround worked effectively, it introduced additional complexities in managing server communication and performance optimization. Another challenge was the performance of Streamlit, as it was often sluggish.