<div align="center">

# ₿ Bitcoin Price Forecasting Portal

**An AI-powered, interactive dashboard for Bitcoin time-series analysis and forecasting.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mazen149-bitcoin-price-forecasting-portal-app-jj4u5l.streamlit.app/)

---

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-4C72B0?style=flat-square)](https://www.statsmodels.org/)
[![Groq AI](https://img.shields.io/badge/Groq_AI-LLaMA_3.1-F55036?style=flat-square&logo=meta&logoColor=white)](https://groq.com/)
[![Kaggle](https://img.shields.io/badge/KaggleHub-Integrated-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://github.com/Kaggle/kagglehub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/website?label=App%20Status&style=flat-square&up_message=online&down_message=offline&url=https%3A%2F%2Fmazen149-bitcoin-price-forecasting-portal-app-jj4u5l.streamlit.app%2F)](https://mazen149-bitcoin-price-forecasting-portal-app-jj4u5l.streamlit.app/)

</div>

---

## 🌟 Overview

The **Bitcoin Price Forecasting Portal** is a production-grade, multi-page Streamlit application that combines advanced time-series analytics with AI-driven insights. It enables users to load, explore, and forecast Bitcoin price data through a sleek, institutional-grade dark-themed dashboard.

**Key Highlights:**
- 📥 **Flexible Data Ingestion** — Upload CSVs, paste GitHub URLs, or one-click fetch from Kaggle
- 📊 **Rich Exploratory Analysis** — Interactive candlestick charts, time-series decomposition, and seasonal heatmaps
- 🔮 **Multi-Model Forecasting** — Holt-Winters, ARIMA, and LSTM (Deep Learning) with future projections
- 🤖 **Strategic AI Insights** — Professional market summaries from a CIO/Strategist perspective, powered by Groq
- ⚡ **Smart Link Injection** — Instant data loading via a one-click example library
- 🎨 **Institutional UI** — Animated "glowing" insight cards with a premium Navy & Gold aesthetic

---

## 🖥️ Live Demo

**Try it now — no installation required:**

<div align="center">

### 🔗 [**Launch the App →**](https://mazen149-bitcoin-price-forecasting-portal-app-jj4u5l.streamlit.app/)

</div>

---

## ✨ Features

### 📥 Data Loader
| Feature | Description |
|---|---|
| **Local Upload** | Drag-and-drop CSV files with automatic parsing and standardization |
| **Remote Fetch** | Paste any GitHub blob URL or direct CSV link |
| **Kaggle Integration** | Enter a dataset slug (e.g., `novandraanugrah/bitcoin-historical-datasets-2018-2024`) for one-click download via KaggleHub |
| **Smart Injection** | One-click library of example links for instant testing and auto-filling |
| **Smart Caching** | Downloaded datasets are cached to disk — same link = instant reload |
| **Auto-Normalization** | Timestamps (Unix/ISO), column names, and **mandatory daily resampling** handled automatically |

### 📊 Explore Data with AI Insights
| Visualization | Description |
|---|---|
| **Candlestick + Volume** | Full OHLCV candlestick chart with volume bars overlay |
| **Time-Series Decomposition** | Trend, seasonality, and residual components (30-day period) |
| **Monthly Seasonality** | Box-plot analysis of returns grouped by calendar month |
| **Log-Return Distribution** | Histogram of daily log returns with a normal distribution fit |
| **Monthly Performance Heatmap** | Year × Month grid of percentage returns with red/green color coding |
| **AI Insight Assistant** | One-click AI summary of market trends, powered by Groq |

### 🔮 Forecasting Engine with AI Insights
| Capability | Description |
|---|---|
| **Holt-Winters** | Exponential smoothing with trend and seasonal components |
| **ARIMA (5,1,0)** | Auto-regressive integrated moving average |
| **LSTM** | **(Default)** PyTorch-based Long Short-Term Memory neural network |
| **Configurable Parameters** | Target column, forecast horizon, train/test split, and confidence interval |
| **Test Data Metrics** | MAE, RMSE, and MAPE on test data predictions |
| **Future Projection** | Visual projection beyond the dataset with confidence bands |
| **AI Forecast Summary** | Plain-language interpretation with automated trend and confidence analysis |

---

## 🏗️ Architecture

```
bitcoin-price-forecasting-portal/
│
├── app.py                     # Main Streamlit entry point & page router
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
│
├── btc_portal/                # Core application modules
│   ├── __init__.py            # Package exports
│   ├── configuration.py       # Forecasting engine UI configuration
│   ├── data_pipeline.py       # CSV parsing, normalization & disk caching
│   ├── forecasting.py         # Holt-Winters, ARIMA, LSTM models
│   ├── ingestion.py           # File upload & remote fetch orchestration
│   ├── llm.py                 # Groq LLM integration for AI insights
│   ├── ui.py                  # Theme, layout, CSS injection & components
│   └── visualization.py       # Plotly chart builders
│
├── .streamlit/
│   ├── config.toml            # Streamlit theme configuration (dark mode)
│   ├── secrets.toml.example   # Template for API keys
│   └── secrets.toml           # Your API keys (git-ignored)
│
└── data_cache/                # Auto-created persistent download cache
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**
- **pip** or [**uv**](https://docs.astral.sh/uv/) (recommended)

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Mazen149/bitcoin-price-forecasting-portal.git
cd bitcoin-price-forecasting-portal
```

**2. Install dependencies:**

<details>
<summary><b>Option A: pip</b></summary>

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

</details>

<details>
<summary><b>Option B: uv (faster)</b></summary>

```bash
uv sync
```

</details>

**3. Configure AI Insights (optional but recommended):**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your [Groq API key](https://console.groq.com/):
```toml
GROQ_API_KEY = "your-groq-api-key-here"
GROQ_MODEL   = "llama-3.1-8b-instant"   # optional, this is the default
```

**4. Launch the app:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📋 Quick-Start Testing

Use these datasets to test the portal end-to-end:

| Source | Link / Slug |
|---|---|
| **Kaggle Link** | `novandraanugrah/bitcoin-historical-datasets-2018-2024` |
| **Kaggle Slug** | `mczielinski/bitcoin-historical-data` |
| **GitHub Blob URL** | `https://github.com/ff137/bitstamp-btcusd-minute-data/blob/main/data/updates/btcusd_bitstamp_1min_latest.csv` |

**Suggested workflow:**
1. Open the **Data Loader** tab
2. Paste a Kaggle slug or GitHub URL and click **Fetch**
3. Switch to **Explore Data** to visualize trends and get AI insights
4. Switch to **Forecasting Engine** to train a model and project future prices

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, custom CSS (dark institutional theme) |
| **Visualization** | Plotly (candlestick, scatter, heatmap, histogram, box) |
| **Data Processing** | Pandas, NumPy |
| **Statistical Models** | Statsmodels (ARIMA, Holt-Winters, seasonal decomposition) |
| **Deep Learning** | PyTorch (LSTM) |
| **ML Utilities** | scikit-learn (scaling, metrics) |
| **AI Insights** | Groq Cloud API (LLaMA 3.1 8B Instant) |
| **Data Sources** | KaggleHub, GitHub raw, direct CSV URLs |

---

## 🎨 Design Philosophy

The dashboard follows an **Institutional Navy & Gold** aesthetic:

- **Color Palette:** Deep navy (`#080E1E`) background, gold (`#FADB5F`) accents, with high-contrast market signals
- **Typography:** Space Grotesk (headings) + JetBrains Mono (data tables)
- **Charts:** Consistent Plotly dark theme with refined typography and gold grids
- **AI Insight Cards:** Bespoke animated "glowing" containers with 3D depth and institutional headers
- **Strategic Persona:** Insights are delivered from the perspective of a **Senior Strategist** or **Chief Investment Officer (CIO)**
- **Notifications:** Custom-styled success/error cards that replace standard Streamlit alerts
- **Layout:** Fully responsive multi-page architecture with persistent sidebar navigation

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq LLM (AI insights) | — |
| `GROQ_MODEL` | LLM model identifier | `llama-3.1-8b-instant` |

These can be set in `.streamlit/secrets.toml` or as system environment variables.

### Streamlit Config

The app ships with a pre-configured `.streamlit/config.toml` that forces dark mode and sets the institutional color palette. No manual theme configuration is needed.

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| **Kaggle download fails** | Ensure Kaggle credentials are configured (`~/.kaggle/kaggle.json`) |
| **AI insights not working** | Set `GROQ_API_KEY` in `.streamlit/secrets.toml` |
| **Memory error on large datasets** | The app automatically handles this via column filtering and daily resampling |
| **Cached data seems stale** | Delete the `data_cache/` directory to force fresh downloads |
| **Model training is slow** | Reduce forecast horizon or try Holt-Winters (fastest model) |

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

</div>
