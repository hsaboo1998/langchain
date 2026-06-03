from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.storage import InMemoryByteStore
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')

# # Document loader. In langchain there are 100's of document loaders to different types and from different sources
loader = PyPDFLoader("https://www.alleycat.org/wp-content/uploads/2018/07/IdentificationBooklet_web.pdf")
document = loader.load() # list of document object per page with metadata
# print(document[1].page_content)
# # print(len(document))

# # split text into meaningful chunks recursively para, newline, spaces, characters to keep context and maintain chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(document)
# # print(min([len(chunk.page_content) for chunk in chunks]), max([len(chunk.page_content) for chunk in chunks]))
# # print(len(chunks))
# # Embeddings and vector database
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
chromavdb = Chroma.from_documents(chunks, embeddings)
# vectors = [embeddings.embed_documents([chunk.page_content]) for chunk in chunks]
# # print(len(vectors))

## similarity search
similar_chunks = chromavdb.similarity_search("How to identify gender of a cat?", k=3)
# print(similar_chunks[0].page_content)

### retrievers

## vectorstore retriever - simple wrapper around vector store
vector_store_retriever = chromavdb.as_retriever(search_type='mmr', search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})

child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

## Parent Document Retriever - use child documents for embedding and sim search and returns parent documents for better context
vector_store = Chroma(collection_name='cats', embedding_function=embeddings)
store=InMemoryByteStore()
parent_document_retriever = ParentDocumentRetriever(vectorstore=vector_store, byte_store=store, 
                                                    child_splitter=child_splitter, parent_splitter=parent_splitter)
parent_document_retriever.add_documents(document)
# print(len(parent_document_retriever.invoke("How to identify gender of a cat?")[0].page_content))

## Self Query Retriever - Uses query constructing llm chain to generate structured query (uses both semantic similarity and metadata filters))
sqr = SelfQueryRetriever.from_llm(
    llm,
    vectorstore=chromavdb,
    document_contents="cat identification guide",
    metadata_field_info=[
        AttributeInfo(
            name="source",
            description="the source url of the document",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="the page number of the document",
            type="integer",
        )
    ],
    structured_query_translator=ChromaTranslator()
    )

## Multi Query Retriever - uses multiple queries generated using llm from different perspectives for a gien query to retrieve diverse set of results
mqr = MultiQueryRetriever.from_llm(retriever=chromavdb.as_retriever(), llm=llm)
# retrieval chain
prompt = PromptTemplate(
    template="""
You are helpful AI assistant for cat lovers. You have the following retrieved documents to answer a questions.
If you don't find any information don't make up the answer. Just say you don't know.
{context}
Don't use any other information.
Question: {question}
""",
input_variables=["context", "question"]
)
chain_type_kwargs = {'prompt': prompt}
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff', # all retrieved documents are concatenated together and passed to llm
    retriever=sqr,
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=False,
    verbose=True
)
print(retrieval_chain.invoke("Which cat has longest coatlength on page 2?"))