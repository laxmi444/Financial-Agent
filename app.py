import streamlit as st

# Streamlit page configuration (MUST be the first command)
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="💹",
    layout="wide"
)

# Import necessary libraries
from langchain.agents import AgentType, initialize_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool, DuckDuckGoSearchRun
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import os

class StockAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze_stock(self, ticker):
        """Analyze stock and return both data and visualization."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Generate the stock visualization
            fig = self.create_stock_chart(ticker)
            
            # Display the chart
            st.subheader(f"{ticker} Stock Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key stock metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Target Price", f"${info.get('targetMeanPrice', 'N/A')}")
            with col3:
                st.metric("Recommendation", info.get('recommendationKey', 'N/A').upper())
            
            return f"I've analyzed {ticker} stock and displayed the chart above. The current price is ${info.get('currentPrice', 'N/A')}, with a target price of ${info.get('targetMeanPrice', 'N/A')}. Analysts currently rate it as {info.get('recommendationKey', 'N/A').upper()}."
            
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"

    def create_stock_chart(self, ticker):
        """Create an interactive stock chart with price and volume."""
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Past one year data
        df = stock.history(start=start_date, end=end_date)
        
        # Create a subplot with two rows: Price and Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Stock Price', 'Trading Volume'), row_heights=[0.7, 0.3])

        # Candlestick chart for price movement
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='OHLC'),
                      row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

        # Moving averages (20-day and 50-day)
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='orange', width=1), name='20-day MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color='blue', width=1), name='50-day MA'), row=1, col=1)
        
        fig.update_layout(title=f'{ticker} Stock Price and Volume', yaxis_title='Price (USD)',
                          yaxis2_title='Volume', xaxis_rangeslider_visible=False, height=800)
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

def main():
    init_page()
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get API key
    api_key = get_api_key()
    
    if not api_key:
        st.error("Please provide a GROQ API key to continue.")
        st.stop()
    
    try:
        # Initialize LLM and tools
        llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, api_key=api_key)
        
        tools = [
            Tool(name="Stock_Analysis", func=st.session_state.analyzer.analyze_stock,
                 description="Analyze a stock and display its chart. Input should be a stock ticker symbol."),
            Tool(name="Web_Search", func=DuckDuckGoSearchRun().run,
                 description="Search the web for current information")
        ]
        
        agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=True, handle_parsing_errors=True)
        
        # Chat input and response handling
        if prompt := st.chat_input("Ask about any stock (e.g., 'Show me AAPL stock chart')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        response = agent.run(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
    
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")

if __name__ == "__main__":
    main()
