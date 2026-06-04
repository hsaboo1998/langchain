from youtube_transcript_api import YouTubeTranscriptApi
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

ytt_api = YouTubeTranscriptApi()
transcripts = ytt_api.list("Qcdnf28K-Gk")
for t in transcripts:
    if t.language_code=="en":
        transcript = t.fetch()
text = " ".join(f"Text:{t.text}. Start:{t.start}" for t in transcript)
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_text(text)
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
faiss_index = FAISS.from_texts(chunks, embedding_model)
template = """
    You are an AI assistant tasked with answering questions about based on the context.
    Do not use timestamps in the answer. Use only the text to answer the question. If you don't know the answer, say you don't know.
    Context: {context}
    Question: {question}
"""
prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=template
)
question = "How to ask are you ok?"
context = faiss_index.similarity_search(question, k=3)
llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
qa_chain = prompt | llm
results = qa_chain.predict(context=context, question=question)
print(results)