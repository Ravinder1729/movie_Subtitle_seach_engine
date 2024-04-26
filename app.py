import streamlit as st
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer,util
st.image(r"C:\Users\ravin\Downloads\Innomatics-Logo1 (1).webp")
st.title("🎥🎬Movie Subtitle Search Engine")


client = chromadb.PersistentClient(path=r"D:\search_engine_subtitles_files")
client.heartbeat()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_collection(name="subtitles_embedded3", embedding_function=sentence_transformer_ef)

model = SentenceTransformer("all-MiniLM-L6-v2")

def encoding_content(x):
    return model.encode(x, normalize_embeddings=True)

def clean_text(text):
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\r\n', '', text)
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'watch any video online with opensubtitles free browser extension osdblinkext', '', text)
    text = text.strip()
    return text

def get_results(query_text):
        

        query_clean = clean_text(query_text)
        query_em = encoding_content(query_clean)

        search_results = collection.query(query_embeddings=query_em.tolist(), n_results=10)
        
        return search_results

user_search = st.text_input("Enter  sub title")
search_results =   get_results(user_search)

if st.button("Search"):
    st.write("Top 10 Results")
    for i, res in enumerate(search_results['metadatas'][0]):
        st.write(f"Search {i+1} : ", res['subtitle_name'])
