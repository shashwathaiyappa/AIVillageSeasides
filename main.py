import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import json

# --- Config ---
api_key = "apikey" # <-- replace this
base_url = "https://integrate.api.nvidia.com/v1"
model = "meta/llama-3.3-70b-instruct"
tavily_api_key = "apikey" # <-- replace this

os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- LLM ---
llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=model,
    temperature=0.2
)

# --- Tavily Search Tool ---
search_tool = TavilySearchResults(max_results=3)

# --- Decision + Answer Function ---
def ask_with_search(user_query: str) -> str:
    # Step 1: Ask LLM if internet search is needed
    decision_prompt = [
        SystemMessage(content=(
            "You are a helpful assistant. Your job is to decide if the user's question "
            "requires an internet search to answer accurately (e.g., current events, recent news, "
            "specific people, live data). "
            "Reply with ONLY a JSON object like: {\"needs_search\": true} or {\"needs_search\": false}. "
            "No explanation, no extra text."
        )),
        HumanMessage(content=user_query)
    ]

    decision_resp = llm.invoke(decision_prompt)
    
    try:
        decision = json.loads(decision_resp.content.strip())
        needs_search = decision.get("needs_search", False)
    except Exception:
        needs_search = False  # fallback: no search

    # Step 2: If search needed, fetch results via Tavily
    if needs_search:
        print("[🔍 Searching the web via Tavily...]")
        search_results = search_tool.invoke(user_query)
        
        # Format results as context
        context = "\n\n".join(
            [f"Source: {r['url']}\n{r['content']}" for r in search_results]
        )
        
        # Step 3: Ask LLM to answer using search results
        answer_messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Use the following search results to answer "
                "the user's question accurately and concisely.\n\n"
                f"Search Results:\n{context}"
            )),
            HumanMessage(content=user_query)
        ]
    else:
        print("[💡 Answering from model knowledge...]")
        answer_messages = [HumanMessage(content=user_query)]

    # Step 4: Get final answer
    final_resp = llm.invoke(answer_messages)
    return final_resp.content


# --- Run ---
query = "tell me about seasides ai village 2026"
answer = ask_with_search(query)
print(answer)
