import streamlit as st

# sstreamlit page configuration (MUST be the first command)
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="💹",
    layout="wide"
)

# import necessary libraries
from langchain.agents import AgentType, initialize_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import os
import re

class StockAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze_stock(self, ticker):
        """Analyze stock and return both data and visualization."""
        try:
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            #generate the stock visualization
            fig = self.create_stock_chart(ticker)
            
            # display the chart
            st.subheader(f"{ticker} Stock Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # display key stock metrics
            cols = st.columns(3)
            with cols[0]:
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                st.metric("Current Price", f"${current_price}" if current_price != 'N/A' else 'N/A')
            with cols[1]:
                target_price = info.get('targetMeanPrice', 'N/A')
                st.metric("Target Price", f"${target_price}" if target_price != 'N/A' else 'N/A')
            with cols[2]:
                recommendation = info.get('recommendationKey', 'N/A')
                st.metric("Recommendation", recommendation.upper() if recommendation != 'N/A' else 'N/A')
            
            # format the response with proper handling for NA values
            current_price_str = f"${current_price}" if current_price != 'N/A' else 'N/A'
            target_price_str = f"${target_price}" if target_price != 'N/A' else 'N/A'
            recommendation_str = recommendation.upper() if recommendation != 'N/A' else 'N/A'
            
            return f"I've analyzed {ticker} stock and displayed the chart above. The current price is {current_price_str}, with a target price of {target_price_str}. Analysts currently rate it as {recommendation_str}."
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            return f"Error analyzing {ticker}: {str(e)}"

    def create_stock_chart(self, ticker):
        """Create an interactive stock chart with price and volume."""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # past one year data
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                st.warning(f"No historical data available for {ticker}")
                # create an empty figure with a message
                fig = go.Figure()
                fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig.update_layout(title=f'{ticker} - No Data Available')
                return fig
            
            # create a subplot with two rows: Price and Volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=('Stock Price', 'Trading Volume'), row_heights=[0.7, 0.3])

            # candlestick chart for price movement
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name='OHLC'),
                          row=1, col=1)

            # volume chart
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

            # moving averages (20-day and 50-day) - only calculate if enough data pointss
            if len(df) >= 20:
                ma20 = df['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='orange', width=1), name='20-day MA'), row=1, col=1)
            
            if len(df) >= 50:
                ma50 = df['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color='blue', width=1), name='50-day MA'), row=1, col=1)
            
            fig.update_layout(
                title=f'{ticker} Stock Price and Volume', 
                yaxis_title='Price (USD)',
                yaxis2_title='Volume', 
                xaxis_rangeslider_visible=False, 
                height=800
            )
            return fig
        except Exception as e:
            st.error(f"Error creating chart for {ticker}: {str(e)}")
            # return an empty figure with error message
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=f'{ticker} - Error Creating Chart')
            return fig

def init_page():
    """Initialize Streamlit UI elements."""
    st.title("Financial AI Assistant 💹")
    st.sidebar.title("About")
    st.sidebar.info(
        """
    This AI assistant can help you with:
    - Stock Analysis with Charts 📊
    - Market Research 📈
    - Financial News 📰
    - Trading Insights 💰
    """
    )

def get_api_key():
    """Retrieve API key from environment or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            if "GROQ_API_KEY" not in st.session_state:
                st.session_state.GROQ_API_KEY = ""
            
            api_key = st.sidebar.text_input("Enter your GROQ API Key:", value=st.session_state.GROQ_API_KEY, type="password")
            if api_key:
                st.session_state.GROQ_API_KEY = api_key
    
    return api_key

def is_stock_query(text):
    """
    Determine if query is asking for stock information.
    Returns (is_stock_query, ticker_symbol)
    """
    # common patterns for stock requests
    stock_patterns = [
        r'show\s+(?:me\s+)?(?:the\s+)?([A-Za-z]+)\s+(?:stock|chart)',  # "show me AAPL stock"
        r'(?:get|fetch|display|analyze)\s+([A-Za-z]+)\s+(?:stock|chart)',  # "analyze AAPL stock"
        r'([A-Za-z]+)\s+(?:stock|chart|price|analysis)',  # "AAPL stock" or "AAPL chart"
        r'how\s+is\s+([A-Za-z]+)\s+(?:doing|performing)',  # "how is AAPL doing"
        r'what\s+is\s+([A-Za-z]+)\s+(?:stock|price)',  # "what is AAPL stock"
    ]
    
    text = text.lower()
    
    for pattern in stock_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ticker = match.group(1).upper()
            return True, ticker
    
    # check for standalone ticker symbols (must be 1-5 letters)
    standalone_ticker = re.search(r'\b([A-Za-z]{1,5})\b', text)
    if standalone_ticker and len(text) < 10:  # Short query with just a ticker
        return True, standalone_ticker.group(1).upper()
    
    return False, None

def is_casual_conversation(text):
    """Determine if the query is casual conversation like greetings or thanks."""
    casual_patterns = [
        r'\b(hi|hello|hey|greetings|howdy)\b',
        r'\b(thanks|thank you|thx|ty|appreciate|gratitude)\b',
        r'\b(bye|goodbye|see you|farewell)\b',
        r'\b(how are you|how\'s it going|what\'s up)\b',
        r'\b(yes|no|maybe|sure|ok|okay|yep|nope)\b'
    ]
    
    text = text.lower()
    
    for pattern in casual_patterns:
        if re.search(pattern, text):
            return True
    
    # very short messages are likely casual
    if len(text.split()) <= 3:
        return True
    
    return False

def get_casual_response(text):
    """Generate appropriate response for casual conversation."""
    text = text.lower()
    
    # greeting patterns
    if re.search(r'\b(hi|hello|hey|greetings|howdy)\b', text):
        return "Hello! I'm your financial assistant. How can I help you with stock analysis or financial information today?"
    
    # thank you patterns
    if re.search(r'\b(thanks|thank you|thx|ty|appreciate|gratitude)\b', text):
        return "You're welcome! Let me know if you need any other financial information or stock analysis."
    
    # goodbye patterns
    if re.search(r'\b(bye|goodbye|see you|farewell)\b', text):
        return "Goodbye! Feel free to return anytime you need financial insights."
    
    # how are you patterns
    if re.search(r'\b(how are you|how\'s it going|what\'s up)\b', text):
        return "I'm doing well, thanks for asking! Ready to help with your financial queries. What would you like to know about stocks today?"
    
    # default casual response
    return "I'm here to help with financial analysis and stock information. What would you like to know about the markets today?"

def process_request(agent, prompt):
    """Process the user request and return the response."""
    # first check if its a casual conversation
    if is_casual_conversation(prompt):
        return get_casual_response(prompt)
    
    # then check if its a stock query
    is_stock, ticker = is_stock_query(prompt)
    if is_stock and ticker:
        try:
            # directly use the stock analyzer for clear stock queries
            return st.session_state.analyzer.analyze_stock(ticker)
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"
    
    # for other queries, use the agent
    try:
        return agent.run(prompt)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    init_page()
    
    # initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    
    # get API key
    api_key = get_api_key()
    
    if not api_key:
        st.warning("Please provide a GROQ API key to continue.")
        st.stop()
    
    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # initialize agent components
    try:
        #initialize LLM and tools
        llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, api_key=api_key)
        
        search_tool = DuckDuckGoSearchRun()
        
        tools = [
            Tool(name="Stock_Analysis", func=st.session_state.analyzer.analyze_stock,
                 description="Analyze a stock and display its chart. Input should be a stock ticker symbol."),
            Tool(name="Web_Search", func=search_tool.run,
                 description="Search the web for current information")
        ]
        
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, 
            handle_parsing_errors=True
        )
        
        #chat input and response handling
        prompt = st.chat_input("Ask about any stock (e.g., 'Show me AAPL stock chart')")
        
        if prompt:
            # check if this is a new query
            if prompt != st.session_state.last_query:
                # update last query
                st.session_state.last_query = prompt
                
                # add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # display assistant message with loading spinner
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        # process the request with improved classification
                        response = process_request(agent, prompt)
                        
                        # display the response
                        st.markdown(response)
                        
                        # add to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")

if __name__ == "__main__":
    main()
