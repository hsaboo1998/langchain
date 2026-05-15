from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# ------------------- prompt from list of messages, output parsing with pydantic model and with_structured_output -------------------------
prompt = ChatPromptTemplate.from_messages(templates=[
 ("system", "You are a helpful assistant"),
 ("user", "Tell me a joke about {topic}")
])
chat = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
class Joke(BaseModel):
    setup: str = Field(description="Question")
    punchline: str = Field(description="Answer")
structured_chat = chat.with_structured_output(Joke)
chain = prompt|structured_chat
result=chain.invoke({"topic": "cats"})
print(result)

# ------------------ Provide inbuilt parsers prompt template ---------------------------------
output_parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_template(template="{format_instructions} List three {topic} jokes",
                                          partial_variables={"format_instructions": output_parser.get_format_instructions()})
chain = prompt|chat|output_parser
result=chain.invoke({"topic": "cats"})
print(result)

# ------------------ converstational memory ---------------------------------
# New version doesn't use conversational memory
