from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv
load_dotenv()

# Document loader. In langchain there are 100's of document loaders to different types and from different sources
loader = PyPDFLoader("https://www.alleycat.org/wp-content/uploads/2018/07/IdentificationBooklet_web.pdf")
document = loader.load() # list of document object per page with metadata
# print(document[1].page_content)
# print(len(document))

# split text into meaningful chunks recursively para, newline, spaces, characters to keep context and maintain chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(document)
# print(min([len(chunk.page_content) for chunk in chunks]), max([len(chunk.page_content) for chunk in chunks]))
# print(len(chunks))
# Embeddings and vector database
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
chromadb = Chroma.from_documents(chunks, embeddings)
vectors = [embeddings.embed_documents([chunk.page_content]) for chunk in chunks]
# print(len(vectors))

# similarity search
similar_chunks = chromadb.similarity_search("How to identify gender of a cat?", k=3)
# print(similar_chunks[0].page_content)

# retrievers
# vectorstore retriever - simple wrapper around vector store
vector_store_retriever = chromadb.as_retriever(search_type='mmr', search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
# parent document retriever - first fetches small chunks and then looks up parent id for larger documents
vector_store = Chroma(collection_name='cats', embedding_function=embeddings)
store=InMemoryByteStore()
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
parent_document_retriever = ParentDocumentRetriever(vectorstore=vector_store, byte_store=store, 
                                                    child_splitter=child_splitter, parent_splitter=parent_splitter)
parent_document_retriever.add_documents(document)
# print(len(parent_document_retriever.invoke("How to identify gender of a cat?")[0].page_content))

# retrieval chain
llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff', # all retrieved documents are concatenated together and passed to llm
    retriever=parent_document_retriever,
    return_source_documents=False
)
# print(retrieval_chain.invoke("Which cat has longest hair?"))