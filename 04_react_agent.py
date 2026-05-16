from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
# ----------------------------------- old style react agent ----------------------------------------------
# Agents use LLM as reasoning engine to identify appropriate set of actions and verify the input and output to llm.
# ReAct(Reasoning + Action) -  Reason, Act, Observation, Repeat, Final Answer
python = PythonREPL()
@tool
def run_python_code(code: str) -> str:
    """Run the following python code and return the output"""
    return python.run(code)
# tools: detailed description of tools, agent_scratchpad: agent's prev reasoning steps agent feeds back its own reasoning
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