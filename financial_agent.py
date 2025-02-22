from langchain.agents import AgentType, initialize_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool, DuckDuckGoSearchRun
import yfinance as yf
import os

# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

def get_stock_info(ticker):
    """Get basic stock information and analyst recommendations"""
    stock = yf.Ticker(ticker)
    info = stock.info
    recommendations = stock.recommendations
    
    return {
        "current_price": info.get('currentPrice', 'N/A'),
        "target_price": info.get('targetMeanPrice', 'N/A'),
        "recommendation": info.get('recommendationKey', 'N/A'),
        "analyst_recommendations": recommendations.tail().to_dict() if not recommendations.empty else "No recommendations available"
    }

def get_stock_news(ticker):
    """Get recent news for a stock"""
    stock = yf.Ticker(ticker)
    news = stock.news
    news_list = []
    
    for item in news[:5]:
        news_dict = {
            "headline": item.get("headline", "No headline available"),
            "publisher": item.get("publisher", "Unknown publisher"),
            "url": item.get("link", "No link available")
        }
        news_list.append(news_dict)
    
    return news_list

# Initialize tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Web_Search",
        func=search.run,
        description="Useful for searching current information and news from the web"
    ),
    Tool(
        name="Stock_Analysis",
        func=get_stock_info,
        description="Get stock information and analyst recommendations. Input should be a stock ticker symbol."
    ),
    Tool(
        name="Stock_News",
        func=get_stock_news,
        description="Get recent news about a stock. Input should be a stock ticker symbol."
    )
]

# Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Example usage
query = "Analyze NVIDIA stock (NVDA). Share the current analyst recommendations and latest news. Also include any relevant market sentiment from recent web searches."
response = agent.run(query)
print(response)