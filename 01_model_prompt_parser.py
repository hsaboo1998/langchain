# Langchain abstraction of openai llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# load env variables
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------- simple qa text generation ----------------------------------
chat = ChatGoogleGenerativeAI(temperature=0, model='gemini-2.5-flash')
template_string = """
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}"""
prompt_template = ChatPromptTemplate.from_template(template_string)

customer_review = """
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""
# customer_message = prompt_template.format_messages(text=customer_review)
# customer_response = chat.invoke(customer_message)
# print(customer_response.content)

# ---------------------------------- Parsing output into a pydantic model ----------------------------------

class Review(BaseModel):
    gift: bool = Field(description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
    delivery_days: int = Field(description="How many days did it take for the product to arrive? If this information is not found, output -1.")
    price_value: list[str] = Field(description="Extract any sentences about the value or price, and output them as a comma separated Python list.")

structured_chat = chat.with_structured_output(Review) # use with_structured_output to parse the response into a pydantic model
prompt = ChatPromptTemplate.from_template(template=template_string)
customer_message = prompt.format_messages(text=customer_review)
customer_response = structured_chat.invoke(customer_message) # schema information is passed behind the scenes
print(customer_response) # output is now a pydantic model instead of a string