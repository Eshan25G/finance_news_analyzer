import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import re
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import yfinance as yf
from newsapi import NewsApiClient
import feedparser
import nltk
from collections import Counter
import asyncio
import aiohttp
from typing import List, Dict, Any

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="📈 Real-Time Finance News Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .news-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #ffc107; font-weight: bold; }
    
    .stMetric > div > div > div > div {
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

class FinanceNewsAnalyzer:
    def __init__(self):
        self.financial_keywords = {
            'positive': ['profit', 'growth', 'increase', 'gain', 'rise', 'bull', 'rally', 'surge', 'boom', 'expansion', 
                        'revenue', 'earnings', 'outperform', 'strong', 'robust', 'upgrade', 'beat', 'exceed'],
            'negative': ['loss', 'decline', 'decrease', 'fall', 'drop', 'bear', 'crash', 'recession', 'bankruptcy', 
                        'debt', 'deficit', 'downturn', 'underperform', 'weak', 'struggling', 'downgrade', 'miss'],
            'neutral': ['merger', 'acquisition', 'announcement', 'report', 'statement', 'market', 'stock', 'shares', 
                       'company', 'financial', 'quarterly', 'annual', 'forecast', 'outlook', 'dividend']
        }
        
        self.stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'BAC']
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the given text"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'Positive'
            color = 'sentiment-positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
            color = 'sentiment-negative'
        else:
            sentiment = 'Neutral'
            color = 'sentiment-neutral'
            
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'color': color,
            'confidence': abs(polarity)
        }
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """Extract financial metrics and keywords from text"""
        text_lower = text.lower()
        
        # Extract numbers with financial context
        price_pattern = r'\$[\d,]+\.?\d*'
        percentage_pattern = r'\d+\.?\d*%'
        
        prices = re.findall(price_pattern, text)
        percentages = re.findall(percentage_pattern, text)
        
        # Extract stock tickers
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        tickers = [t for t in potential_tickers if t in self.stock_tickers]
        
        # Count keyword categories
        pos_count = sum(1 for word in self.financial_keywords['positive'] if word in text_lower)
        neg_count = sum(1 for word in self.financial_keywords['negative'] if word in text_lower)
        neu_count = sum(1 for word in self.financial_keywords['neutral'] if word in text_lower)
        
        return {
            'prices': prices,
            'percentages': percentages,
            'tickers': tickers,
            'positive_keywords': pos_count,
            'negative_keywords': neg_count,
            'neutral_keywords': neu_count,
            'total_financial_words': pos_count + neg_count + neu_count
        }
    
    def calculate_risk_score(self, sentiment_data: Dict, metrics_data: Dict) -> Dict[str, Any]:
        """Calculate risk score based on sentiment and content"""
        base_score = 50  # Neutral starting point
        
        # Adjust based on sentiment
        if sentiment_data['sentiment'] == 'Positive':
            base_score -= 20
        elif sentiment_data['sentiment'] == 'Negative':
            base_score += 30
            
        # Adjust based on keyword balance
        if metrics_data['negative_keywords'] > metrics_data['positive_keywords']:
            base_score += 15
        elif metrics_data['positive_keywords'] > metrics_data['negative_keywords']:
            base_score -= 10
            
        # Ensure score is within bounds
        risk_score = max(0, min(100, base_score))
        
        if risk_score < 30:
            risk_level = 'Low'
            risk_color = '#28a745'
        elif risk_score < 70:
            risk_level = 'Medium'
            risk_color = '#ffc107'
        else:
            risk_level = 'High'
            risk_color = '#dc3545'
            
        return {
            'score': risk_score,
            'level': risk_level,
            'color': risk_color
        }

def get_rss_news(rss_urls: List[str]) -> List[Dict]:
    """Fetch news from RSS feeds"""
    news_articles = []
    
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # Limit to 5 articles per feed
                news_articles.append({
                    'title': entry.title,
                    'description': entry.get('description', ''),
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown')
                })
        except Exception as e:
            st.error(f"Error fetching from {url}: {str(e)}")
    
    return news_articles

def get_stock_data(ticker: str) -> Dict:
    """Get real-time stock data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")
        info = stock.info
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            change = current_price - open_price
            change_percent = (change / open_price) * 100
            
            return {
                'symbol': ticker,
                'current_price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': hist['Volume'].iloc[-1],
                'high': hist['High'].max(),
                'low': hist['Low'].min(),
                'company_name': info.get('shortName', ticker)
            }
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
    
    return None

def main():
    # Initialize analyzer
    analyzer = FinanceNewsAnalyzer()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Real-Time Finance News Analyzer</h1>
        <p>Analyze financial news sentiment and extract key insights in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    
    # Data sources
    st.sidebar.subheader("📊 Data Sources")
    use_rss = st.sidebar.checkbox("Enable RSS News Feeds", value=True)
    use_manual = st.sidebar.checkbox("Manual Text Analysis", value=True)
    
    # RSS feeds
    rss_feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.marketwatch.com/rss/topstories",
        "https://feeds.bloomberg.com/markets/news.rss"
    ]
    
    # Main content area
    if use_manual:
        st.header("📝 Manual News Analysis")
        
        # Text input
        news_text = st.text_area(
            "Enter financial news text:",
            height=150,
            placeholder="Paste your financial news article here for analysis..."
        )
        
        # Example button
        if st.button("Load Example Text"):
            news_text = """Apple Inc. (AAPL) reported strong quarterly earnings today, with revenue jumping 12% year-over-year to $85.5 billion. The tech giant beat analyst expectations on both revenue and earnings per share, with iPhone sales showing robust growth despite supply chain challenges. CEO Tim Cook highlighted the company's strong performance in services revenue, which grew 15% to $19.2 billion. The stock surged 8% in after-hours trading following the announcement."""
            st.rerun()
        
        if news_text and st.button("🔍 Analyze Text"):
            # Analyze the text
            sentiment_data = analyzer.analyze_sentiment(news_text)
            metrics_data = analyzer.extract_financial_metrics(news_text)
            risk_data = analyzer.calculate_risk_score(sentiment_data, metrics_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Sentiment",
                    sentiment_data['sentiment'],
                    f"{sentiment_data['polarity']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Risk Level",
                    risk_data['level'],
                    f"{risk_data['score']}/100"
                )
            
            with col3:
                st.metric(
                    "Financial Keywords",
                    metrics_data['total_financial_words'],
                    f"Pos: {metrics_data['positive_keywords']}, Neg: {metrics_data['negative_keywords']}"
                )
            
            # Detailed analysis
            st.subheader("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Extracted Information:**")
                if metrics_data['tickers']:
                    st.write(f"🏢 **Tickers:** {', '.join(metrics_data['tickers'])}")
                if metrics_data['prices']:
                    st.write(f"💰 **Prices:** {', '.join(metrics_data['prices'])}")
                if metrics_data['percentages']:
                    st.write(f"📈 **Percentages:** {', '.join(metrics_data['percentages'])}")
            
            with col2:
                st.write("**Sentiment Details:**")
                st.write(f"📊 **Polarity:** {sentiment_data['polarity']:.2f}")
                st.write(f"🎯 **Subjectivity:** {sentiment_data['subjectivity']:.2f}")
                st.write(f"🔒 **Confidence:** {sentiment_data['confidence']:.2f}")
    
    # RSS News Feed Analysis
    if use_rss:
        st.header("📰 Live News Feed Analysis")
        
        # Create placeholder for live data
        news_placeholder = st.empty()
        
        # Fetch and display news
        with st.spinner("Fetching latest financial news..."):
            news_articles = get_rss_news(rss_feeds)
        
        if news_articles:
            st.success(f"Found {len(news_articles)} recent articles")
            
            # Analyze each article
            for i, article in enumerate(news_articles[:10]):  # Show top 10
                with st.expander(f"📄 {article['title'][:100]}..."):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Published:** {article['published']}")
                    st.write(f"**Link:** {article['link']}")
                    
                    # Analyze article
                    full_text = f"{article['title']} {article['description']}"
                    if len(full_text) > 50:
                        sentiment = analyzer.analyze_sentiment(full_text)
                        metrics = analyzer.extract_financial_metrics(full_text)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Sentiment:** {sentiment['sentiment']}")
                        with col2:
                            st.write(f"**Polarity:** {sentiment['polarity']:.2f}")
                        with col3:
                            st.write(f"**Tickers:** {', '.join(metrics['tickers']) if metrics['tickers'] else 'None'}")
    
    # Stock prices section
    st.header("💹 Real-Time Stock Prices")
    
    selected_tickers = st.multiselect(
        "Select stocks to monitor:",
        analyzer.stock_tickers,
        default=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    )
    
    if selected_tickers:
        stock_cols = st.columns(min(len(selected_tickers), 4))
        
        for i, ticker in enumerate(selected_tickers):
            with stock_cols[i % 4]:
                stock_data = get_stock_data(ticker)
                if stock_data:
                    delta_color = "normal" if stock_data['change'] >= 0 else "inverse"
                    st.metric(
                        f"{ticker}",
                        f"${stock_data['current_price']:.2f}",
                        f"{stock_data['change']:+.2f} ({stock_data['change_percent']:+.1f}%)",
                        delta_color=delta_color
                    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>📈 Real-Time Finance News Analyzer | Built with Streamlit</p>
        <p>Data sources: Yahoo Finance, RSS Feeds | Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
