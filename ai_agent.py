from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY= os.getenv('GROQ_API_KEY')
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")

#Step 2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(model='gpt-4o-mini')
groq_llm = ChatGroq(model='llama-3.3-70b-versatile')

search_tavily_tool = TavilySearchResults(max_results=2)

#Step 3: Setup AI Agent with Search Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Defining the AI Agent Function
system_prompt = "I am an AI assistant that can help you with your queries. How can I help you today?"

# A below function to generate responses from the AI agent.

# Parameters:

# llm_id: Specifies which LLM model to use.

# query: The user’s input question/query.

# allow_search: Boolean (True/False) that determines whether web search is enabled.

# system_prompt: Custom instructions for AI behavior.

# provider: Specifies whether to use "Groq" or "OpenAI".

def get_response_from_ai_agent(llm_id, query, allowed_search, system_prompt, provider):   
    if provider == 'Groq':
        llm = ChatGroq(model=llm_id)
    elif provider == 'OpenAI':
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearchResults] if allowed_search else []
    agent = create_react_agent(
            model = llm,
            tools = tools,
            state_modifier= system_prompt
    )
    state= {"messages": query}
    response = agent.invoke(state)
    messages= response.get("messages")
    #print(messages)
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]