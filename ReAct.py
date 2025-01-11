import os
from typing import Annotated
from datetime import datetime
from tavily import TavilyClient
from typing import Annotated, Literal
from pdb import set_trace
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, register_function
# from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from mem0 import MemoryClient
# import agentops
# agentops.init(api_key="4d498c5f-8355-4a39-8e86-4d4b0013fc89")

from mem0 import MemoryClient
memory_client = MemoryClient(api_key="m0-vuVSKfdnHTdCAJ1m4QFmnz8PKgpE0Yz3lbbHBMMp")

config_list = [
    {
    "model": "gpt-4o-mini",
    #  "model": "gpt-4o", 
     "api_key": os.environ["OPENAI_API_KEY"], 
     "base_url": os.environ.get("OPENAI_API_BASE")},
    # {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"], "base_url": os.environ.get("OPENAI_API_BASE")},
]
# You can also use the following method to load the config list from a file or environment variable.
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    # return tavily.get_search_context(query=query, search_depth="advanced")
    return tavily.get_search_context(query=query)

def get_current_time():
    "return the current time in the format of YYYY-mm-DD HH:MM:SS"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

relevant_memories = memory_client.search(
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?", 
    user_id="customer_service_bot")
flatten_relevant_memories = "\n".join([m["memory"] for m in relevant_memories])
# Answer the following questions as best you can.
# If the questions is related to current time, you should always use the tools to get the current year, date or current time first.
AssistPrompt = f"""
You already know something: {flatten_relevant_memories}.
Solve the given task step by step. Use the following format:

Question: the question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
(you can do only one action after you think about and output previous steps, one action one step at a time, after you do the action, you should output Observation as below)
Observation: the result of the action
... (the above process can repeat multiple times)

FinalThought: I now know the final answer
Final Answer: the final answer to the original input question
"""


# AssistPrompt = """
# Solve the given task by following Thought, Action and Observation for every step.
# Thought can reason about the current situation and next steps to follow, 
# Action must be called one by one and only after you think about and output previous steps.
# Observation is what results are achieved after taking Action. 

# current year is 2024.
# """

# # for math problem
# AssistPrompt = """
# Answer the following math problem as best you can. You have access to tools provided.
# If the questions is related to current time, you should always use the tools to get the current year, date or current time first.
# Solve it step by step. Use the following format:

# Question: the question you must answer
# Thought: you should always think about what to do
# Action: the action to take
# Action Input: the input to the action
# (you can do the action after you think about and output previous steps, one action one step at a time, after you do the action, you should output Observation as below)
# Observation: the result of the action
# ... (the above process can repeat multiple times)

# FinalThought: I now know the final answer
# Final Answer: the final answer to the original input question
# """

# Define the ReAct prompt message. Assuming a "question" field is present in the context


# def react_prompt_message(sender, recipient, context):
#     return ReAct_prompt.format(input=context["question"])

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int: 
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")
    
# Setting up code executor.
os.makedirs("coding", exist_ok=True)
# Use docker executor for running code in a container if you have docker installed.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    code_execution_config={"executor": code_executor},
)

assistant = AssistantAgent(
    name="Assistant",
    system_message=AssistPrompt+"\n Reply TERMINATE when the task is done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

# Register the search tool.
register_function(
    search_tool,
    caller=assistant,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

# Register the time tool.
register_function(
    get_current_time,
    caller=assistant,
    executor=user_proxy,
    name="get_current_time",
    description="Get the current date and time in the format of YYYY-mm-DD HH:MM:SS",
)

# Register the calculator function to the two agents.
# register_function(
#     calculator,
#     caller=assistant,  # The assistant agent can suggest calls to the calculator.
#     executor=user_proxy,  # The user proxy agent can execute the calculator calls.
#     name="calculator",  # By default, the function name is used as the tool name.
#     description="A simple calculator",  # A description of the tool.
# )

task = "What is the result of super bowl 2024?"
# Cache LLM responses. To get different responses, change the cache_seed value.
with Cache.disk(cache_seed=43) as cache:
    user_proxy.initiate_chat(
        assistant,
        message=task,
        cache=cache,
    )
