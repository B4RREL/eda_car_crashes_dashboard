# 🚗 Car Crash Analytics Dashboard

An interactive dashboard analyzing U.S. car crash statistics using Streamlit and Google's Gemini AI for dynamic data exploration and visualization.

## 📊 Overview

This project provides a comprehensive analysis of car crashes across U.S. states, focusing on key factors like alcohol involvement, speeding, and insurance implications. The dashboard combines traditional data visualization with AI-powered analysis for an intuitive user experience.

## 🌟 Key Features

- **Interactive Dashboard**:
  - Real-time data visualization
  - Multiple view options (Dashboard, AI Chat, Key Metrics)
  - Responsive design with modern UI

- **AI-Powered Analysis**:
  - Natural language queries
  - Dynamic chart generation
  - Context-aware responses
  - State name expansion (e.g., CA → California)

- **Visualization Types**:
  - Bar charts for rankings
  - Choropleth maps for geographic analysis
  - Scatter plots for correlations
  - Histograms for distributions
  - Advanced visualizations (sunburst, parallel categories)

## 📈 Key Insights

1. **Alcohol Impact**
   - Strongest correlation with total crashes
   - Most consistent predictor across states
   - Clear geographic patterns in involvement rates

2. **Speeding Patterns**
   - Secondary factor compared to alcohol
   - Variable impact across different states
   - Less predictive of total crash numbers

3. **Insurance Relationships**
   - Premium pricing shows weak correlation with crash rates
   - Complex risk profiles command higher premiums
   - State-by-state variation in coverage costs

4. **Geographic Trends**
   - Identifiable regional hotspots
   - State-specific risk patterns
   - Varied enforcement effectiveness

5. **Risk Profiles**
   - Mixed-risk states pay highest premiums
   - Clear patterns in high-risk behaviors
   - State-specific intervention opportunities

## 🗃️ Dataset Overview

The dataset (`car_crashes.csv`) contains the following features:
- `total`: Total number of crashes per state
- `speeding`: Speeding-related crashes
- `alcohol`: Alcohol-related crashes
- `not_distracted`: Non-distracted driving crashes
- `no_previous`: Crashes by drivers with no previous incidents
- `ins_premium`: Average insurance premium
- `ins_losses`: Average insurance losses
- `abbrev`: State abbreviation

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **AI Integration**: Google Gemini AI
- **Styling**: Custom CSS

## 🚀 Getting Started

1. **Installation**
   ```bash
   git clone https://github.com/B4RREL/eda_car_crashes_dashboard.git
   cd car-crashes-project
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   ```bash
   # Create .env file with:
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Running the App**
   ```bash
   streamlit run app.py
   ```

## 📱 Usage Guide

### Dashboard Section
- View 5 key insights with interactive visualizations
- Hover over charts for detailed information
- Use filters to customize views

### AI Chat Section
- Ask natural language questions about the data
- Request specific visualizations
- Get detailed analysis and comparisons

### Key Metrics Section
- View state-by-state rankings
- Compare performance metrics
- Analyze risk factors


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Your Name - [b4rrel.root@gmail.com](mailto: b4rrel.root@gmail.com)

Project Link: [https://github.com/B4RREL/eda_car_crashes_dashboard.git](https://github.com/B4RREL/eda_car_crashes_dashboard.git)

Streamlit Link(Try testing it here): [https://edacarcrashesdashboard-e3bpufffph4pa4ophsnvuo.streamlit.app/](https://edacarcrashesdashboard-e3bpufffph4pa4ophsnvuo.streamlit.app/)

## 🙏 Acknowledgments

- Data source: U.S. Car Crash Statistics
- Streamlit for the amazing framework
- Google for Gemini AI capabilities