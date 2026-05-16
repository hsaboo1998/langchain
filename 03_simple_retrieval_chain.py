from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
import os

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

# # Embeddings and vector database
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# chromadb = Chroma.from_documents(chunks, embeddings)
# vectors = [embeddings.embed_documents([chunk.page_content]) for chunk in chunks]
# # print(len(vectors))

# # similarity search
# similar_chunks = chromadb.similarity_search("How to identify gender of a cat?", k=3)
# # print(similar_chunks[0].page_content)

# # retrievers
# # vectorstore retriever - simple wrapper around vector store
# vector_store_retriever = chromadb.as_retriever(search_type='mmr', search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
# # parent document retriever - first fetches small chunks and then looks up parent id for larger documents
# vector_store = Chroma(collection_name='cats', embedding_function=embeddings)
# store=InMemoryByteStore()
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# parent_document_retriever = ParentDocumentRetriever(vectorstore=vector_store, byte_store=store, 
#                                                     child_splitter=child_splitter, parent_splitter=parent_splitter)
# parent_document_retriever.add_documents(document)
# # print(len(parent_document_retriever.invoke("How to identify gender of a cat?")[0].page_content))

# # retrieval chain
# retrieval_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type='stuff', # all retrieved documents are concatenated together and passed to llm
#     retriever=parent_document_retriever,
#     return_source_documents=False
# )
# print(retrieval_chain.invoke("Which cat has longest hair?"))

# ----------------------------------- old style react agent ----------------------------------------------
# Agents use LLM as reasoning engine to identify appropriate set of actions and verify the input and output to llm.
# ReAct(Reasoning + Action) -  Reason, Act, Observation, Repeat, Final Answer
python = PythonREPL()
@tool
def run_python_code(code: str) -> str:
    """Run the following python code and return the output"""
    return python.run(code)
prompt_template = """You are an agent who has access to the following tools:
{tools}
The available tools are: {tool_names}
To use a tool, please use the following format:
```
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
```
After you use a tool, the observation will be provided to you:
```
Observation: result of the tool
```
Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
Use only mathematical operations and no python functions
When you have the final answer, respond in this format:
```
Thought: I know the answer
Final Answer: the final answer to the original query
```
Remember, when using the Python Calculator tool, the input must be valid Python code.
Begin!
Question: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat",
    temperature=0
)
react_agent = create_react_agent(llm, [run_python_code], prompt)
# Agent Executor
# Handles orchestration between agent reasoning and tool execution.
# Sends initial query to agent, parses agents response to identify tool calls,
# executes the tool, feeds result back to agent until final answer,
# handles parsing errors and implement retry logic for failed executors
# maintain conversation history using memory
# can enforce max iterations and timeout to prevent infinite loops
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=[run_python_code],
    handle_parsing_errors=True,
    max_iterations=50,
    max_execution_time=120,
    verbose=True
)
print(agent_executor.invoke({"input":"What is the standard deviation in these comma separated values: 1,2,3,4,5?"}))