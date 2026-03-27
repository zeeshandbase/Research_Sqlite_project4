# ---------------- INSTALL ----------------
# uv add langchain langchain-community langchain-ollama ddgs arxiv wikipedia

# ---------------- IMPORTS ----------------
import sqlite3

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    ArxivAPIWrapper
)

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT,
    user_query TEXT,
    ai_response TEXT
)
""")
conn.commit()

# ---------------- SAVE FUNCTION ----------------
def save_chat(tool, user_query, ai_response):
    """Save chat history into SQLite database."""
    cursor.execute(
        "INSERT INTO chats (tool, user_query, ai_response) VALUES (?, ?, ?)",
        (tool, user_query, ai_response)
    )
    conn.commit()

# ---------------- CUSTOM TOOL ----------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's always sunny in {city}!"

# ---------------- API TOOLS ----------------
wiki_api = WikipediaAPIWrapper()

@tool
def search_wikipedia(query: str):
    """Search Wikipedia and return a summary of the topic."""
    return wiki_api.run(query)

search_api = DuckDuckGoSearchAPIWrapper()

@tool
def search_duckduckgo(query: str):
    """Search the web using DuckDuckGo and return results."""
    return search_api.run(query)

arxiv_api = ArxivAPIWrapper()

@tool
def search_arxiv(query: str):
    """Search arXiv for research papers related to the query."""
    return arxiv_api.run(query)

# ---------------- MODEL ----------------
model = ChatOllama(model="qwen3.5:397b-cloud")

# ---------------- TOOL MAP ----------------
tool_dict = {
    1: ("Weather", get_weather),
    2: ("Wikipedia", search_wikipedia),
    3: ("DuckDuckGo", search_duckduckgo),
    4: ("arXiv", search_arxiv)
}

# ---------------- AGENT ----------------
agent = create_agent(
    model=model,
    tools=[get_weather, search_wikipedia, search_duckduckgo, search_arxiv],
    system_prompt="You are a helpful AI assistant that can use tools when needed.",
)

# ---------------- LOOP ----------------
while True:
    print("\nChoose a tool by number:")
    for num, (name, _) in tool_dict.items():
        print(f"{num}. {name}")
    print("0. Exit")

    choice = input("Enter your choice: ")

    if not choice.isdigit():
        print("Please enter a number!")
        continue

    choice = int(choice)

    if choice == 0:
        print("Exiting...")
        break

    if choice not in tool_dict:
        print("Invalid choice!")
        continue

    tool_name, tool_func = tool_dict[choice]

    query = input(f"Enter your question for {tool_name}: ")

    # Agent call
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    ai_text = response["messages"][-1].content

    print("\n--- Agent Response ---")
    print(ai_text)
    print("----------------------")

    # Save chat
    save_chat(tool_name, query, ai_text)

# Close DB
conn.close()