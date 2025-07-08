import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="📈 Real-Time Finance News Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    .risk-low { background: #d4edda; padding: 0.5rem; border-radius: 5px; color: #155724; }
    .risk-medium { background: #fff3cd; padding: 0.5rem; border-radius: 5px; color: #856404; }
    .risk-high { background: #f8d7da; padding: 0.5rem; border-radius: 5px; color: #721c24; }
</style>
""", unsafe_allow_html=True)

class SimpleFinanceAnalyzer:
    def __init__(self):
        self.positive_words = [
            'profit', 'growth', 'increase', 'gain', 'rise', 'bull', 'rally', 'surge', 'boom', 
            'expansion', 'revenue', 'earnings', 'outperform', 'strong', 'robust', 'upgrade', 
            'beat', 'exceed', 'success', 'breakthrough', 'soar', 'jump', 'climb', 'advance'
        ]
        
        self.negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'drop', 'bear', 'crash', 'recession', 
            'bankruptcy', 'debt', 'deficit', 'downturn', 'underperform', 'weak', 'struggling', 
            'downgrade', 'miss', 'failure', 'plunge', 'tumble', 'slide', 'collapse', 'slump'
        ]
        
        self.neutral_words = [
            'merger', 'acquisition', 'announcement', 'report', 'statement', 'market', 'stock', 
            'shares', 'company', 'financial', 'quarterly', 'annual', 'forecast', 'outlook', 
            'dividend', 'analyst', 'estimate', 'guidance', 'conference', 'meeting'
        ]
        
        self.stock_tickers = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'BAC',
            'WMT', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'INTC'
        ]
    
    def simple_sentiment_analysis(self, text: str) -> dict:
        """Simple sentiment analysis using keyword counting"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_words if word in text_lower)
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            sentiment = 'Positive'
            polarity = min(0.8, positive_count / max(total_words, 1) * 10)
        elif negative_count > positive_count:
            sentiment = 'Negative'
            polarity = -min(0.8, negative_count / max(total_words, 1) * 10)
        else:
            sentiment = 'Neutral'
            polarity = 0.0
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'confidence': abs(polarity)
        }
    
    def extract_financial_data(self, text: str) -> dict:
        """Extract financial information from text"""
        # Extract prices
        price_pattern = r'\$[\d,]+\.?\d*'
        prices = re.findall(price_pattern, text)
        
        # Extract percentages
        percentage_pattern = r'\d+\.?\d*%'
        percentages = re.findall(percentage_pattern, text)
        
        # Extract stock tickers
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        tickers = [t for t in potential_tickers if t in self.stock_tickers]
        
        # Extract numbers
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, text)
        
        return {
            'prices': prices[:10],  # Limit results
            'percentages': percentages[:10],
            'tickers': list(set(tickers)),
            'numbers': numbers[:10],
            'has_financial_data': len(prices) > 0 or len(percentages) > 0 or len(tickers) > 0
        }
    
    def calculate_risk_score(self, sentiment_data: dict, financial_data: dict) -> dict:
        """Calculate risk score"""
        base_score = 50
        
        # Sentiment impact
        if sentiment_data['sentiment'] == 'Positive':
            base_score -= 15
        elif sentiment_data['sentiment'] == 'Negative':
            base_score += 25
        
        # Financial data impact
        if financial_data['has_financial_data']:
            base_score -= 5  # Having financial data reduces uncertainty
        
        # Word balance impact
        if sentiment_data['negative_count'] > sentiment_data['positive_count']:
            base_score += 10
        elif sentiment_data['positive_count'] > sentiment_data['negative_count']:
            base_score -= 10
        
        risk_score = max(0, min(100, base_score))
        
        if risk_score < 35:
            risk_level = 'Low'
            risk_class = 'risk-low'
        elif risk_score < 65:
            risk_level = 'Medium'
            risk_class = 'risk-medium'
        else:
            risk_level = 'High'
            risk_class = 'risk-high'
        
        return {
            'score': risk_score,
            'level': risk_level,
            'class': risk_class
        }

def get_yahoo_finance_rss():
    """Get news from Yahoo Finance RSS"""
    try:
        url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Parse XML manually
            root = ET.fromstring(response.content)
            articles = []
            
            for item in root.findall('.//item')[:10]:  # Get first 10 items
                title = item.find('title')
                description = item.find('description')
                link = item.find('link')
                pubDate = item.find('pubDate')
                
                articles.append({
                    'title': title.text if title is not None else 'No title',
                    'description': description.text if description is not None else 'No description',
                    'link': link.text if link is not None else '#',
                    'pubDate': pubDate.text if pubDate is not None else 'No date',
                    'source': 'Yahoo Finance'
                })
            
            return articles
        
    except Exception as e:
        st.error(f"Error fetching Yahoo Finance RSS: {str(e)}")
        return []

def get_mock_stock_data(ticker: str) -> dict:
    """Generate mock stock data for demonstration"""
    import random
    
    # Generate realistic stock data
    base_price = random.uniform(50, 500)
    change = random.uniform(-10, 10)
    change_percent = (change / base_price) * 100
    
    return {
        'symbol': ticker,
        'current_price': base_price,
        'change': change,
        'change_percent': change_percent,
        'volume': random.randint(1000000, 10000000),
        'high': base_price + random.uniform(0, 5),
        'low': base_price - random.uniform(0, 5)
    }

def main():
    # Initialize analyzer
    analyzer = SimpleFinanceAnalyzer()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Real-Time Finance News Analyzer</h1>
        <p>Analyze financial news sentiment and extract key insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    # Analysis options
    st.sidebar.subheader("📊 Analysis Options")
    show_manual = st.sidebar.checkbox("Manual Text Analysis", value=True)
    show_news = st.sidebar.checkbox("Live News Feed", value=True)
    show_stocks = st.sidebar.checkbox("Stock Monitor", value=True)
    
    # Manual Analysis Section
    if show_manual:
        st.header("📝 Manual News Analysis")
        
        # Sample texts for quick testing
        sample_texts = {
            "Positive Example": "Apple Inc. reported strong quarterly earnings today, with revenue jumping 12% year-over-year to $85.5 billion. The tech giant beat analyst expectations, with iPhone sales showing robust growth. The stock surged 8% in after-hours trading.",
            "Negative Example": "Tesla shares plummeted 15% after the company missed quarterly delivery targets. The electric vehicle maker reported weaker than expected sales, citing supply chain challenges and increased competition. Analysts downgraded the stock.",
            "Neutral Example": "Microsoft announced a new quarterly dividend of $0.68 per share, maintaining its current payout ratio. The company also confirmed its annual shareholders meeting will be held next month. No major changes to guidance were announced."
        }
        
        selected_sample = st.selectbox("Quick Examples:", ["Select an example..."] + list(sample_texts.keys()))
        
        if selected_sample != "Select an example...":
            news_text = sample_texts[selected_sample]
        else:
            news_text = ""
        
        news_text = st.text_area(
            "Enter financial news text:",
            value=news_text,
            height=150,
            placeholder="Paste your financial news article here for analysis..."
        )
        
        if news_text and st.button("🔍 Analyze Text", type="primary"):
            # Perform analysis
            sentiment_data = analyzer.simple_sentiment_analysis(news_text)
            financial_data = analyzer.extract_financial_data(news_text)
            risk_data = analyzer.calculate_risk_score(sentiment_data, financial_data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sentiment", sentiment_data['sentiment'], f"{sentiment_data['polarity']:.2f}")
            
            with col2:
                st.metric("Risk Level", risk_data['level'], f"{risk_data['score']}/100")
            
            with col3:
                st.metric("Positive Words", sentiment_data['positive_count'])
            
            with col4:
                st.metric("Negative Words", sentiment_data['negative_count'])
            
            # Detailed results
            st.subheader("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Extracted Financial Data:**")
                if financial_data['tickers']:
                    st.write(f"**Tickers:** {', '.join(financial_data['tickers'])}")
                if financial_data['prices']:
                    st.write(f"**Prices:** {', '.join(financial_data['prices'])}")
                if financial_data['percentages']:
                    st.write(f"**Percentages:** {', '.join(financial_data['percentages'])}")
                
                if not financial_data['has_financial_data']:
                    st.write("*No specific financial data detected*")
            
            with col2:
                st.write("**📈 Sentiment Analysis:**")
                st.write(f"**Sentiment:** {sentiment_data['sentiment']}")
                st.write(f"**Polarity:** {sentiment_data['polarity']:.2f}")
                st.write(f"**Confidence:** {sentiment_data['confidence']:.2f}")
                
                # Risk assessment
                st.markdown(f"""
                <div class="{risk_data['class']}">
                    <strong>Risk Assessment:</strong> {risk_data['level']} ({risk_data['score']}/100)
                </div>
                """, unsafe_allow_html=True)
    
    # Live News Feed
    if show_news:
        st.header("📰 Live News Feed")
        
        if st.button("🔄 Refresh News"):
            with st.spinner("Fetching latest financial news..."):
                articles = get_yahoo_finance_rss()
            
            if articles:
                st.success(f"Found {len(articles)} recent articles")
                
                for i, article in enumerate(articles):
                    with st.expander(f"📄 {article['title'][:80]}..."):
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Published:** {article['pubDate']}")
                        st.write(f"**Description:** {article['description']}")
                        st.write(f"**Link:** [Read more]({article['link']})")
                        
                        # Quick analysis
                        full_text = f"{article['title']} {article['description']}"
                        if len(full_text) > 50:
                            sentiment = analyzer.simple_sentiment_analysis(full_text)
                            financial = analyzer.extract_financial_data(full_text)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Sentiment:** {sentiment['sentiment']}")
                            with col2:
                                st.write(f"**Polarity:** {sentiment['polarity']:.2f}")
                            with col3:
                                if financial['tickers']:
                                    st.write(f"**Tickers:** {', '.join(financial['tickers'])}")
                                else:
                                    st.write("**Tickers:** None detected")
            else:
                st.warning("No articles found. Try refreshing or check your internet connection.")
    
    # Stock Monitor
    if show_stocks:
        st.header("💹 Stock Monitor")
        st.info("📊 Live stock data integration coming soon! Currently showing demo data.")
        
        selected_tickers = st.multiselect(
            "Select stocks to monitor:",
            analyzer.stock_tickers,
            default=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        )
        
        if selected_tickers:
            cols = st.columns(min(len(selected_tickers), 4))
            
            for i, ticker in enumerate(selected_tickers):
                with cols[i % 4]:
                    stock_data = get_mock_stock_data(ticker)
                    
                    # Determine color based on change
                    delta_color = "normal" if stock_data['change'] >= 0 else "inverse"
                    
                    st.metric(
                        f"{ticker}",
                        f"${stock_data['current_price']:.2f}",
                        f"{stock_data['change']:+.2f} ({stock_data['change_percent']:+.1f}%)",
                        delta_color=delta_color
                    )
    
    # Performance Dashboard
    st.header("📊 Analysis Dashboard")
    
    # Create sample data for visualization
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Add current analysis to history (if available)
    if st.button("📈 Generate Sample Analytics"):
        # Create sample sentiment data over time
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        sentiment_scores = np.random.normal(0, 0.3, len(dates))
        
        df = pd.DataFrame({
            'Date': dates,
            'Sentiment_Score': sentiment_scores,
            'Risk_Score': 50 + sentiment_scores * 30 + np.random.normal(0, 10, len(dates))
        })
        
        # Plot sentiment over time
        fig = px.line(df, x='Date', y='Sentiment_Score', 
                     title='Sentiment Analysis Over Time',
                     labels={'Sentiment_Score': 'Sentiment Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk score distribution
        fig2 = px.histogram(df, x='Risk_Score', nbins=20,
                           title='Risk Score Distribution',
                           labels={'Risk_Score': 'Risk Score'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p>📈 Real-Time Finance News Analyzer | Built with Streamlit</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
