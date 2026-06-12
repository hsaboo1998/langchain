from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat",
    temperature=0
)
df = pd.read_csv('student-mat.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
print(agent.invoke("Generate a bar char to to plot gender count?")['output'])