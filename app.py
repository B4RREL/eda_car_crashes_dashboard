# app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time

# =====================
# Page Configuration
# =====================
st.set_page_config(
    page_title="Car Crash Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# Custom CSS for Modern Design
# =====================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modern color scheme and typography */
    .main {
        padding-top: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        color: #1e293b;
    }

    .insight-number {
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
    }

    /* Chat styling */
    .chat-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
    }

    /* Cards for insights */
    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        color: #1e293b;
    }

    /* Plotly chart containers */
    .plot-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Data Loading with Caching
# =====================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv("car_crashes.csv")
    return df

@st.cache_data
def get_state_map():
    """Cache state abbreviation mapping"""
    return {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
        "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
        "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
        "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
        "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
        "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
    }

# =====================
# Optimized AI Setup with Caching
# =====================
@st.cache_resource
def initialize_ai():
    """Initialize and cache AI components"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found in environment variables")
        return None, None

    genai.configure(api_key=api_key)

    # Initialize LLM with caching
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=1000
    )

    model = genai.GenerativeModel("gemini-2.5-flash")

    return llm, model

# Load data and initialize AI
df = load_data()
state_map = get_state_map()
llm, model = initialize_ai()

# =====================
# Sidebar Navigation
# =====================
with st.sidebar:
    st.markdown("### 🚗 Car Crash Analytics")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate to:",
        ["📊 Dashboard", "🤖 AI Chat", "📈 Key Metrics", "ℹ️ About"],
        index=0
    )

    st.markdown("---")

    # Quick Stats in Sidebar
    st.markdown("#### Quick Stats")

    total_crashes = df['total'].sum()
    avg_alcohol_rate = (df['alcohol'].sum() / df['total'].sum() * 100)
    avg_speeding_rate = (df['speeding'].sum() / df['total'].sum() * 100)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{total_crashes:,.0f}</p>
        <p class="metric-label">Total Crashes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{avg_alcohol_rate:.1f}%</p>
        <p class="metric-label">Alcohol Related</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{avg_speeding_rate:.1f}%</p>
        <p class="metric-label">Speeding Related</p>
    </div>
    """, unsafe_allow_html=True)

# =====================
# Main Content Based on Navigation
# =====================

if page == "📊 Dashboard":
    # Header
    st.markdown("""
    # 🚗 Car Crash Analysis: Insights & Story

    This dashboard presents **5 key insights** about U.S. car crashes with interactive visualizations
    and data-driven conclusions.
    """)

    # =====================================================
    # Insight 1: Alcohol is the most consistent driver
    # =====================================================
    st.markdown("""
    <div class="section-header">
        <div class="insight-number">1</div>
        <h2 style="margin: 0;">Alcohol is the most consistent driver of crash totals</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            fig1 = px.scatter(
                df, x="alcohol", y="total",
                trendline="ols",
                labels={"alcohol": "Alcohol Involvement", "total": "Crash Total"},
                title="Alcohol vs Total Crashes",
                template="plotly_white"
            )
            fig1.update_traces(marker=dict(size=8, opacity=0.7, color='#ef4444'))
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            correlation = df[['alcohol', 'total']].corr().iloc[0, 1]
            st.markdown(f"""
            <div class="insight-card">
                <h4>📊 Key Finding</h4>
                <p><strong>Correlation:</strong> {correlation:.3f}</p>
                <p>Strong positive relationship shows alcohol involvement is highly predictive of total crash numbers across states.</p>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # Insight 2: Speeding matters, but less strongly
    # =====================================================
    st.markdown("""
    <div class="section-header">
        <div class="insight-number">2</div>
        <h2 style="margin: 0;">Speeding matters, but doesn't scale as strongly</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig2 = px.scatter(
            df, x="speeding", y="total",
            trendline="ols",
            labels={"speeding": "Speeding Involvement", "total": "Crash Total"},
            title="Speeding vs Total Crashes",
            template="plotly_white"
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.7, color='#f59e0b'))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        speed_correlation = df[['speeding', 'total']].corr().iloc[0, 1]
        st.markdown(f"""
        <div class="insight-card">
            <h4>📊 Key Finding</h4>
            <p><strong>Correlation:</strong> {speed_correlation:.3f}</p>
            <p>Positive trend but weaker than alcohol. Some high-speeding states don't necessarily have extreme crash totals.</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # Insight 3: Insurance pricing correlation
    # =====================================================
    st.markdown("""
    <div class="section-header">
        <div class="insight-number">3</div>
        <h2 style="margin: 0;">Insurance pricing is not correlated with crash totals</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig3 = px.scatter(
            df, x="ins_premium", y="total",
            labels={"ins_premium": "Insurance Premium ($)", "total": "Crash Total"},
            title="Insurance Premium vs Crash Totals",
            template="plotly_white"
        )
        fig3.update_traces(marker=dict(size=8, opacity=0.7, color='#8b5cf6'))
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        ins_correlation = df[['ins_premium', 'total']].corr().iloc[0, 1]
        st.markdown(f"""
        <div class="insight-card">
            <h4>📊 Key Finding</h4>
            <p><strong>Correlation:</strong> {ins_correlation:.3f}</p>
            <p>No clear trend. Insurance premiums are influenced more by economic and market factors than accident frequency.</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # Insight 4: Geographic outliers
    # =====================================================
    st.markdown("""
    <div class="section-header">
        <div class="insight-number">4</div>
        <h2 style="margin: 0;">Geographic patterns reveal alcohol hotspots</h2>
    </div>
    """, unsafe_allow_html=True)

    fig4 = px.choropleth(
        df,
        locations="abbrev",
        locationmode="USA-states",
        color="alcohol",
        scope="usa",
        color_continuous_scale="Reds",
        title="Alcohol-Related Crashes by State",
        labels={"alcohol": "Alcohol Crashes"}
    )
    fig4.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Find top alcohol states
    top_alcohol = df.nlargest(3, 'alcohol')[['abbrev', 'alcohol']].values
    st.markdown(f"""
    <div class="insight-card">
        <h4>🎯 Hotspot States</h4>
        <p><strong>Top 3:</strong> {top_alcohol[0][0]} ({top_alcohol[0][1]}), {top_alcohol[1][0]} ({top_alcohol[1][1]}), {top_alcohol[2][0]} ({top_alcohol[2][1]}) crashes</p>
        <p>These states show disproportionately high alcohol involvement and may benefit from targeted interventions.</p>
        <p>Policy changes and increased enforcement could help mitigate these issues.</p>
        <ul>
            <li>Found these outliers could be legitimate by checking the difference between mean and median alcohol rates.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # Insight 5: Risk profiles and insurance
    # =====================================================
    st.markdown("""
    <div class="section-header">
        <div class="insight-number">5</div>
        <h2 style="margin: 0;">Mixed-profile states pay highest insurance premiums</h2>
    </div>
    """, unsafe_allow_html=True)

    # Calculate risk profiles (cached computation)
    @st.cache_data
    def calculate_risk_profiles(df):
        df_copy = df.copy()
        df_copy["alcohol_rate"] = df_copy["alcohol"] / df_copy["total"]
        df_copy["speeding_rate"] = df_copy["speeding"] / df_copy["total"]

        gap = (df_copy['speeding_rate'] - df_copy['alcohol_rate']).abs()
        df_copy['dominant_risk'] = np.select(
            [gap < 0.05, df_copy['speeding_rate'] > df_copy['alcohol_rate']],
            ['Mixed', 'Speeding'],
            default='Alcohol'
        )
        return df_copy

    df_with_risk = calculate_risk_profiles(df)
    avg_cost = df_with_risk.groupby('dominant_risk')[['total','ins_premium']].mean().reset_index()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig5 = px.bar(
            avg_cost,
            x="dominant_risk",
            y="ins_premium",
            title="Average Insurance Premium by Risk Profile",
            labels={"ins_premium": "Avg Insurance Premium ($)", "dominant_risk": "Risk Type"},
            color="dominant_risk",
            template="plotly_white"
        )
        fig5.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter",
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        mixed_premium = avg_cost[avg_cost['dominant_risk'] == 'Mixed']['ins_premium'].values[0]
        st.markdown(f"""
        <div class="insight-card">
            <h4>💰 Premium Analysis</h4>
            <p><strong>Mixed Risk Premium:</strong> ${mixed_premium:.0f}</p>
            <p>Insurers charge higher premiums for complex risk profiles where neither alcohol nor speeding clearly dominates.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🤖 AI Chat":
    st.markdown("# 🤖 AI Data Assistant")
    st.markdown("Ask me anything about the car crash data! I can help you explore patterns, calculate statistics, and provide insights with **interactive charts**.")
    st.markdown("💡 **Try asking**: *'Show me top 5 states for alcohol crashes'* or *'Create a bar plot for alcohol top 5 states'*")

    # Initialize session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    def get_dataset_info():
        info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "description": {
                "total": "Total number of car crashes per state",
                "speeding": "Number of crashes involving speeding",
                "alcohol": "Number of crashes involving alcohol",
                "not_distracted": "Number of crashes where driver was not distracted",
                "no_previous": "Number of crashes by drivers with no previous violations",
                "ins_premium": "Average car insurance premium in dollars",
                "ins_losses": "Average insurance losses per state",
                "abbrev": "State abbreviation (AL, CA, TX, etc.)"
            },
            "sample_data": df.head().to_dict()
        }
        return info

    # Add this function just after the get_dataset_info() function (around line 696)
    def get_dashboard_guide():
        return """Let me guide you through our Car Crash Analytics Dashboard! 🚗

        1. **Navigation** (Left Sidebar 👈)
        - 📊 **Dashboard**: View 5 key insights with interactive visualizations
        - 🤖 **AI Chat**: Ask questions about the data (you're here!)
        - 📈 **Key Metrics**: See important statistics and state rankings
        - ℹ️ **About**: Learn about the dashboard's features
        - 📈 **Key Metrics**: See important statistics and state rankings
        - ℹ️ **About**: Learn about the dashboard's features

        2. **Quick Stats** (Always visible in sidebar)
        - Total crashes across all states
        - Percentage of alcohol-related crashes
        - Percentage of speeding-related crashes

        3. **How to Use This Dashboard:**

        👉 **For Data Exploration:**
        - Use the Dashboard page to view pre-built insights
        - Hover over charts for detailed information
        - Click and drag on charts to zoom in

        👉 **For Custom Analysis (AI Chat):**
        - Ask questions like:
            • "Show me top 5 states with highest crashes"
            • "Create a bar plot for alcohol-related crashes"
            • "What's the correlation between speeding and total crashes?"

        👉 **For State Rankings:**
        - Check the Key Metrics page for top and bottom performers
        - View state-by-state comparisons
        - See average insurance premiums

        4. **Pro Tips:**
        - Use natural language for questions
        - Ask for specific visualizations
        - Compare different factors
        - Look for patterns across states

        Need anything specific? Just ask! 😊"""

    def should_show_chart(question):
        """Determine if the question should include a chart"""
        chart_keywords = [
            'show', 'chart', 'graph', 'plot', 'visualize', 'compare', 'bar plot', 'bar chart',
            'top', 'highest', 'lowest', 'distribution', 'correlation', 'histogram',
            'relationship', 'trend', 'pattern', 'versus', 'vs', 'map', 'geographic'
        ]
        return any(keyword in question.lower() for keyword in chart_keywords)

    def create_chart_for_question(question):
        question_lower = question.lower()

        try:
            # Top/Highest states questions
            if any(word in question_lower for word in ['top', 'highest', 'most']) and any(word in question_lower for word in ['bar', 'chart', 'plot', 'show']):
                if 'alcohol' in question_lower:
                    # Extract number if mentioned (default to 10)
                    import re
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))  # Don't exceed available data

                    top_data = df.nlargest(n, 'alcohol')[['abbrev', 'alcohol']]
                    fig = px.bar(top_data, x='abbrev', y='alcohol',
                               title=f'Top {n} States by Alcohol-Related Crashes',
                               labels={'abbrev': 'State', 'alcohol': 'Alcohol Crashes'},
                               color='alcohol',
                               color_continuous_scale='Reds')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the top {n} states for alcohol-related crashes."

                elif 'speeding' in question_lower:
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))

                    top_data = df.nlargest(n, 'speeding')[['abbrev', 'speeding']]
                    fig = px.bar(top_data, x='abbrev', y='speeding',
                               title=f'Top {n} States by Speeding-Related Crashes',
                               labels={'abbrev': 'State', 'speeding': 'Speeding Crashes'},
                               color='speeding',
                               color_continuous_scale='Oranges')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the top {n} states for speeding-related crashes."

                elif any(word in question_lower for word in ['crashes', 'total']):
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))

                    top_data = df.nlargest(n, 'total')[['abbrev', 'total']]
                    fig = px.bar(top_data, x='abbrev', y='total',
                               title=f'Top {n} States by Total Crashes',
                               labels={'abbrev': 'State', 'total': 'Total Crashes'},
                               color='total',
                               color_continuous_scale='Blues')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the top {n} states for total crashes."

                elif any(word in question_lower for word in ['insurance', 'premium']):
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))

                    top_data = df.nlargest(n, 'ins_premium')[['abbrev', 'ins_premium']]
                    fig = px.bar(top_data, x='abbrev', y='ins_premium',
                               title=f'Top {n} States by Insurance Premium',
                               labels={'abbrev': 'State', 'ins_premium': 'Insurance Premium ($)'},
                               color='ins_premium',
                               color_continuous_scale='Purples')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the top {n} states for insurance premiums."

            # Bottom/Lowest states questions
            elif any(word in question_lower for word in ['bottom', 'lowest', 'least', 'safest']):
                if 'alcohol' in question_lower:
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))

                    bottom_data = df.nsmallest(n, 'alcohol')[['abbrev', 'alcohol']]
                    fig = px.bar(bottom_data, x='abbrev', y='alcohol',
                               title=f'Bottom {n} States by Alcohol-Related Crashes',
                               labels={'abbrev': 'State', 'alcohol': 'Alcohol Crashes'},
                               color='alcohol',
                               color_continuous_scale='Greens')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the bottom {n} states for alcohol-related crashes."

                elif any(word in question_lower for word in ['crashes', 'total']):
                    numbers = re.findall(r'\d+', question)
                    n = int(numbers[0]) if numbers else 10
                    n = min(n, len(df))

                    bottom_data = df.nsmallest(n, 'total')[['abbrev', 'total']]
                    fig = px.bar(bottom_data, x='abbrev', y='total',
                               title=f'Bottom {n} States by Total Crashes',
                               labels={'abbrev': 'State', 'total': 'Total Crashes'},
                               color='total',
                               color_continuous_scale='Greens')
                    fig.update_layout(template="plotly_white")
                    return fig, f"Here's a bar chart showing the bottom {n} states for total crashes."

            # Correlation/Relationship questions
            elif any(word in question_lower for word in ['correlation', 'relationship', 'versus', 'vs', 'scatter']):
                if 'alcohol' in question_lower and 'total' in question_lower:
                    fig = px.scatter(df, x='alcohol', y='total', hover_data=['abbrev'],
                                   title='Alcohol vs Total Crashes',
                                   labels={'alcohol': 'Alcohol Crashes', 'total': 'Total Crashes'},
                                   trendline='ols')
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a scatter plot showing the relationship between alcohol-related crashes and total crashes."

                elif 'speeding' in question_lower and 'total' in question_lower:
                    fig = px.scatter(df, x='speeding', y='total', hover_data=['abbrev'],
                                   title='Speeding vs Total Crashes',
                                   labels={'speeding': 'Speeding Crashes', 'total': 'Total Crashes'},
                                   trendline='ols')
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a scatter plot showing the relationship between speeding-related crashes and total crashes."

                elif 'insurance' in question_lower and 'crashes' in question_lower:
                    fig = px.scatter(df, x='ins_premium', y='total', hover_data=['abbrev'],
                                   title='Insurance Premium vs Total Crashes',
                                   labels={'ins_premium': 'Insurance Premium ($)', 'total': 'Total Crashes'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a scatter plot showing the relationship between insurance premiums and total crashes."

            # Distribution/Histogram questions
            elif any(word in question_lower for word in ['distribution', 'histogram', 'spread']):
                if 'alcohol' in question_lower:
                    fig = px.histogram(df, x='alcohol', nbins=20,
                                     title='Distribution of Alcohol-Related Crashes',
                                     labels={'alcohol': 'Alcohol Crashes', 'count': 'Number of States'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a histogram showing the distribution of alcohol-related crashes across states."

                elif 'speeding' in question_lower:
                    fig = px.histogram(df, x='speeding', nbins=20,
                                     title='Distribution of Speeding-Related Crashes',
                                     labels={'speeding': 'Speeding Crashes', 'count': 'Number of States'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a histogram showing the distribution of speeding-related crashes across states."

                elif 'insurance' in question_lower:
                    fig = px.histogram(df, x='ins_premium', nbins=20,
                                     title='Distribution of Insurance Premiums',
                                     labels={'ins_premium': 'Insurance Premium ($)', 'count': 'Number of States'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a histogram showing the distribution of insurance premiums across states."

            # Map questions
            elif any(word in question_lower for word in ['map', 'geographic', 'geography']):
                if 'alcohol' in question_lower:
                    fig = px.choropleth(df, locations='abbrev', locationmode='USA-states',
                                      color='alcohol', scope='usa', color_continuous_scale='Reds',
                                      title='Alcohol-Related Crashes by State',
                                      labels={'alcohol': 'Alcohol Crashes'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a geographic map showing alcohol-related crashes across states."

                elif any(word in question_lower for word in ['total', 'crashes']):
                    fig = px.choropleth(df, locations='abbrev', locationmode='USA-states',
                                      color='total', scope='usa', color_continuous_scale='Blues',
                                      title='Total Crashes by State',
                                      labels={'total': 'Total Crashes'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a geographic map showing total crashes across states."

                elif 'speeding' in question_lower:
                    fig = px.choropleth(df, locations='abbrev', locationmode='USA-states',
                                      color='speeding', scope='usa', color_continuous_scale='Oranges',
                                      title='Speeding-Related Crashes by State',
                                      labels={'speeding': 'Speeding Crashes'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a geographic map showing speeding-related crashes across states."

            # Compare/Comparison questions
            elif 'compare' in question_lower or 'comparison' in question_lower:
                # Multi-metric comparison
                if 'speeding' in question_lower and 'alcohol' in question_lower and not any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest']):
                    fig = px.scatter(df, x='alcohol', y='speeding',
                                size='total', hover_data=['abbrev', 'ins_premium'],
                                title='Multi-Factor Comparison: Alcohol vs Speeding (Size = Total Crashes)',
                                labels={'alcohol': 'Alcohol Crashes', 'speeding': 'Speeding Crashes'})
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a bubble chart comparing alcohol vs speeding crashes, with bubble size representing total crashes."

            elif any(word in question_lower for word in ['advanced', 'detailed', 'complex']):

                # 1. Multi-metric Sunburst Chart
                if 'breakdown' in question_lower or 'composition' in question_lower:
                    df_temp = df.copy()
                    df_temp['other'] = df_temp['total'] - df_temp['speeding'] - df_temp['alcohol']

                    # Prepare data for sunburst
                    sunburst_data = pd.DataFrame({
                        'State': df_temp['abbrev'].repeat(3),
                        'Type': ['Speeding', 'Alcohol', 'Other'] * len(df_temp),
                        'Value': pd.concat([df_temp['speeding'], df_temp['alcohol'], df_temp['other']])
                    })

                    fig = px.sunburst(
                        sunburst_data,
                        path=['State', 'Type'],
                        values='Value',
                        title='Crash Type Composition by State',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's an interactive sunburst chart showing the breakdown of crash types for each state."

                # 2. Animated Time Comparison (if time data available)
                elif 'trend' in question_lower or 'over time' in question_lower:
                    df_trends = df.sort_values('total', ascending=True)
                    df_trends['frames'] = range(len(df_trends))

                    fig = px.bar(
                        df_trends,
                        x='abbrev',
                        y=['speeding', 'alcohol'],
                        animation_frame='frames',
                        barmode='group',
                        title='Progressive Comparison of Speeding vs Alcohol Crashes',
                        labels={'value': 'Number of Crashes', 'variable': 'Crash Type'},
                        color_discrete_map={'speeding': '#f59e0b', 'alcohol': '#ef4444'}
                    )
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's an animated chart comparing speeding and alcohol crashes across states."

                # 3. Advanced Parallel Categories
                elif 'categories' in question_lower or 'parallel' in question_lower:
                    df_cat = df.copy()
                    # Create categories for each metric
                    df_cat['crash_level'] = pd.qcut(df_cat['total'], q=3, labels=['Low', 'Medium', 'High'])
                    df_cat['alcohol_level'] = pd.qcut(df_cat['alcohol'], q=3, labels=['Low', 'Medium', 'High'])
                    df_cat['premium_level'] = pd.qcut(df_cat['ins_premium'], q=3, labels=['Low', 'Medium', 'High'])

                    fig = px.parallel_categories(
                        df_cat,
                        dimensions=['crash_level', 'alcohol_level', 'premium_level'],
                        color='total',
                        labels={'crash_level': 'Crash Level',
                            'alcohol_level': 'Alcohol Level',
                            'premium_level': 'Insurance Premium'},
                        title='Multi-dimensional Pattern Analysis'
                    )
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a parallel categories plot showing relationships between crash levels, alcohol involvement, and insurance premiums."

                # 4. Bubble Matrix
                elif 'matrix' in question_lower or 'bubble' in question_lower:
                    fig = px.scatter(
                        df,
                        x='alcohol',
                        y='speeding',
                        size='total',
                        color='ins_premium',
                        hover_name='abbrev',
                        size_max=60,
                        title='Multi-factor Bubble Matrix',
                        labels={
                            'alcohol': 'Alcohol Crashes',
                            'speeding': 'Speeding Crashes',
                            'total': 'Total Crashes',
                            'ins_premium': 'Insurance Premium ($)'
                        }
                    )
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a bubble matrix showing relationships between multiple factors. Bubble size represents total crashes, color shows insurance premium."

                # 5. Advanced Heatmap
                elif 'heatmap' in question_lower or 'correlation matrix' in question_lower:
                    # Calculate correlation matrix
                    corr_matrix = df[['total', 'speeding', 'alcohol', 'ins_premium', 'ins_losses']].corr()

                    fig = px.imshow(
                        corr_matrix,
                        text=np.round(corr_matrix, 2),
                        aspect='auto',
                        title='Correlation Matrix Heatmap',
                        color_continuous_scale='RdBu',
                        labels={'color': 'Correlation'}
                    )
                    fig.update_layout(template="plotly_white")
                    return fig, "Here's a detailed correlation heatmap showing relationships between all metrics. Blue indicates positive correlation, red indicates negative."
            else:
                    try:
                        # Create a more specific prompt for chart generation
                        chart_prompt = f"""
                        Generate ONLY Python code for a Plotly chart. Question: "{question}"

                        Available data (DataFrame 'df'):
                        - total: Total crashes per state
                        - speeding: Speeding crashes per state
                        - alcohol: Alcohol crashes per state
                        - ins_premium: Insurance premiums
                        - ins_losses: Insurance losses
                        - abbrev: State abbreviations (e.g., CA, NY)

                        Requirements:
                        1. Use only plotly.express (px) or plotly.graph_objects (go)
                        2. Include data preparation code if needed (sorting, filtering, etc.)
                        3. For comparisons, use appropriate chart types (grouped bars, scatter, etc.)
                        4. Always set template="plotly_white"
                        5. Include proper titles, labels, and color schemes
                        6. Store final figure in 'fig' variable

                        Example format for complex comparison:
                        ```python
                        # Prepare data
                        top_states = df.nlargest(5, 'total')[['abbrev', 'total', 'alcohol']]
                        bottom_states = df.nsmallest(5, 'total')[['abbrev', 'total', 'alcohol']]
                        compare_df = pd.concat([top_states, bottom_states])

                        # Create figure
                        fig = px.bar(compare_df, x='abbrev', y=['total', 'alcohol'], barmode='group')
                        fig.update_layout(template="plotly_white")
                        ```

                        Return ONLY the code, no explanations.
                        """

                        # Get code from model
                        response = model.generate_content(chart_prompt)
                        generated_code = response.text.strip()

                        # Add safety checks and common imports
                        safe_globals = {
                            'pd': pd,
                            'px': px,
                            'go': go,
                            'np': np,
                            'df': df.copy()  # Use a copy for safety
                        }

                        # Execute the generated code
                        try:
                            exec(generated_code, safe_globals)
                            if 'fig' in safe_globals:
                                fig = safe_globals['fig']

                                # Get description
                                desc_prompt = f"Describe this visualization in one sentence, mentioning the type of chart and what it shows. Question: {question}"
                                desc_response = model.generate_content(desc_prompt)
                                description = desc_response.text.strip()

                                return fig, description
                        except Exception as code_error:
                            print(f"Code execution error: {str(code_error)}")
                            return None, None

                    except Exception as e:
                        print(f"Chart generation error: {str(e)}")
                        return None, None
        except Exception as e:
            print(f"Unexpected error in chart creation: {str(e)}")
        return None, None

    # Display chat history with charts
    for i, turn in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            # Display chart if it exists in history
            if "chart" in turn and turn["chart"] is not None:
                st.plotly_chart(turn["chart"], use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Ask me about the crash data..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            if llm and model:
                with st.spinner("Analyzing your question and creating visualizations..."):
                    try:
                        # Check if we should create a chart
                        show_chart = should_show_chart(prompt)
                        chart = None
                        chart_description = ""

                        if show_chart:
                            chart, chart_description = create_chart_for_question(prompt)

                        # Check for common questions that don't need pandas agent
                        dataset_info = get_dataset_info()

                        # Handle specific question types without pandas agent
                        if any(keyword in prompt.lower() for keyword in ['features', 'columns', 'variables', 'what data', 'dataset info']):
                            # Direct response for dataset questions
                            direct_prompt = f"""
                            You are a data analyst. The user asks: "{prompt}"

                            Here's the car crash dataset information:
                            - Dataset shape: {dataset_info['shape'][0]} states, {dataset_info['shape'][1]} columns
                            - Columns: {', '.join(dataset_info['columns'])}

                            Column descriptions:
                            {chr(10).join([f"• {col}: {desc}" for col, desc in dataset_info['description'].items()])}

                            Sample data (first row): {dataset_info['sample_data']}

                            Provide a helpful response about the dataset features and what they represent.
                            Focus on what makes this data useful for car crash analysis.
                            """

                            response = model.generate_content(direct_prompt)
                            humanized = response.text
                        elif any(keyword in prompt.lower() for keyword in ['help', 'guide', 'how to use', 'instructions',
                        'tutorial', 'how do i', 'what can you do', 'what can i ask', 'how does this work']):
                            try:
                                # Get base guide content
                                base_guide = get_dashboard_guide()

                                # Create a prompt for the model to personalize the guide
                                guide_prompt = f"""
                                The user asks: "{prompt}"

                                Here's our dashboard guide:
                                {base_guide}

                                Please personalize this guide based on the user's specific question.
                                Focus on the most relevant sections and add examples that address their needs.
                                Keep the response friendly and structured.
                                """

                                # Get personalized response from model
                                response = model.generate_content(guide_prompt)
                                humanized = response.text

                                # Save to history
                                st.session_state.chat_history.append({
                                    "question": prompt,
                                    "answer": humanized,
                                    "chart": None
                                })
                            except Exception as e:
                                # Fallback to static guide if model fails
                                st.markdown(get_dashboard_guide())
                                st.session_state.chat_history.append({
                                    "question": prompt,
                                    "answer": get_dashboard_guide(),
                                    "chart": None
                                })
                        else:
                            # Use pandas agent for complex queries with better error handling
                            agent = create_pandas_dataframe_agent(
                                llm, df,
                                verbose=False,
                                allow_dangerous_code=True,
                                max_iterations=3,
                                handle_parsing_errors=True,
                                early_stopping_method="generate"
                            )

                            # Get raw result
                            raw_result = agent.run(prompt)

                            # Build conversation context
                            recent_history = st.session_state.chat_history[-2:] if len(st.session_state.chat_history) > 2 else st.session_state.chat_history
                            history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in recent_history])

                            # Include chart info in response
                            chart_info = f" {chart_description}" if chart else ""

                            # Humanize response with better prompt
                            explain_prompt = f"""
                            You are a friendly data analyst discussing car crash data.

                            Recent conversation context:
                            {history_text}

                            User's question: "{prompt}"
                            Analysis result: {raw_result}
                            Chart created: {chart_info}

                            Provide a clear, conversational response that:
                            - Explains the findings in plain English
                            - Uses the conversation context if relevant
                            - Converts state codes to full names when mentioned
                            - Stays focused and concise
                            - Uses proper formatting for readability
                            {"- Mentions the chart/visualization I created" if chart else ""}

                            Keep your response under 200 words and be helpful.
                            """

                            response = model.generate_content(explain_prompt)
                            humanized = response.text

                        # Expand state abbreviations
                        for abbr, full in state_map.items():
                            patterns_to_replace = [
                                (f" {abbr} ", f" {full} ({abbr}) "),
                                (f" {abbr},", f" {full} ({abbr}),"),
                                (f" {abbr}.", f" {full} ({abbr})."),
                                (f"'{abbr}'", f"'{full} ({abbr})'"),
                                (f'"{abbr}"', f'"{full} ({abbr})"')
                            ]
                            for pattern, replacement in patterns_to_replace:
                                humanized = humanized.replace(pattern, replacement)

                        # Display text response
                        st.markdown(humanized)

                        # Display chart if created
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

                        # Save to history with chart
                        st.session_state.chat_history.append({
                            "question": prompt,
                            "answer": humanized,
                            "chart": chart
                        })

                    except Exception as e:
                        # Fallback to direct AI response
                        error_msg = str(e)
                        if "parsing error" in error_msg.lower() or "output parsing" in error_msg.lower():
                            st.warning("🔄 Let me try a different approach...")

                            # Try to create chart anyway
                            show_chart = should_show_chart(prompt)
                            chart = None
                            if show_chart:
                                chart, _ = create_chart_for_question(prompt)

                            # Fallback direct response
                            fallback_prompt = f"""
                            You are analyzing a car crash dataset with these columns:
                            {', '.join(get_dataset_info()['columns'])}

                            The user asks: "{prompt}"

                            Based on your knowledge of data analysis and the column names,
                            provide a helpful response about car crash data analysis.
                            Keep it practical and informative.
                            """

                            try:
                                response = model.generate_content(fallback_prompt)
                                fallback_answer = response.text
                                st.markdown(fallback_answer)

                                # Show chart if created
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)

                                # Save fallback response
                                st.session_state.chat_history.append({
                                    "question": prompt,
                                    "answer": fallback_answer,
                                    "chart": chart
                                })

                            except Exception as fallback_error:
                                st.error(f"⚠️ I'm having trouble processing your question: {str(fallback_error)}")
                                st.markdown("Please try rephrasing your question or ask something more specific about the car crash data.")
                        else:
                            st.error(f"⚠️ Sorry, I encountered an error: {str(e)}")
                            st.markdown("Try rephrasing your question or ask something more specific about the data.")
            else:
                st.error("⚠️ AI assistant is not available. Please check your API key configuration.")

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Key Metrics":
    st.markdown("# 📈 Key Metrics Dashboard")

    # Create metrics grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total States",
            value=len(df),
            delta=None
        )

    with col2:
        st.metric(
            label="Avg Crashes per State",
            value=f"{df['total'].mean():.0f}",
            delta=f"{df['total'].std():.0f} std dev"
        )

    with col3:
        st.metric(
            label="Highest Risk State",
            value=df.loc[df['total'].idxmax(), 'abbrev'],
            delta=f"{df['total'].max()} crashes"
        )

    with col4:
        st.metric(
            label="Avg Insurance Premium",
            value=f"${df['ins_premium'].mean():.0f}",
            delta=f"${df['ins_premium'].std():.0f} std dev"
        )

    # Detailed breakdown
    st.markdown("## 📊 Detailed Breakdown")

    # Top performers table
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Highest Risk States")
        top_risk = df.nlargest(10, 'total')[['abbrev', 'total', 'alcohol', 'speeding']]
        st.dataframe(top_risk, use_container_width=True)

    with col2:
        st.markdown("### 🟢 Lowest Risk States")
        low_risk = df.nsmallest(10, 'total')[['abbrev', 'total', 'alcohol', 'speeding']]
        st.dataframe(low_risk, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Dashboard")

    st.markdown("""
    ## 🎯 Purpose
    This interactive dashboard analyzes U.S. car crash data to identify key patterns and insights
    that can inform policy decisions and safety interventions.

    ## 📊 Data Sources
    - **Dataset**: Car crash statistics by U.S. state
    - **Features**: Total crashes, alcohol involvement, speeding, insurance data
    - **Scope**: All 50 U.S. states + D.C.

    ## 🔍 Key Insights
    1. **Alcohol Correlation**: Strongest predictor of total crashes
    2. **Speeding Impact**: Significant but weaker relationship
    3. **Insurance Disconnect**: Premiums not tied to crash frequency
    4. **Geographic Patterns**: Clear regional hotspots
    5. **Risk Complexity**: Mixed profiles cost more to insure

    ## 🤖 AI Features
    - **Natural Language Queries**: Ask questions about the data
    - **Contextual Responses**: Maintains conversation context
    - **Data Analysis**: Performs real-time calculations
    - **State Name Expansion**: Converts abbreviations automatically

    ## 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **Visualizations**: Plotly Express
    - **AI**: Google Gemini 2.5 Flash
    - **Data Processing**: Pandas, NumPy
    - **Deployment**: Optimized with caching and session management

    ## 📱 Usage Tips
    - Use the sidebar navigation to switch between sections
    - Try asking the AI specific questions about states or metrics
    - Hover over charts for detailed information
    - All visualizations are interactive and responsive
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    🚗 Car Crash Analytics Dashboard | Built with Streamlit & Google AI
</div>
""", unsafe_allow_html=True)