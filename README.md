# Implied Volatility Surface Viewer  

A Python-based web app that visualizes the **implied volatility surface** for equity options using real-time data from Yahoo Finance. The app computes implied volatilities with the Black-Scholes model and presents them on an interactive 3D surface.  

## 🌐 Live Demo  
[Launch the App on Streamlit](https://implied-volatility-surface-anthony.streamlit.app/)  

## ✨ Features  
- **Real-Time Financial Data**: Pulls live options, risk-free rate, and yield data via Yahoo Finance.  
- **Interactive 3D Surface**: Visualizes implied volatility against strike price (or moneyness) and time to expiration.  
- **Customizable Parameters**:  
  - Risk-free rate & dividend yield  
  - Ticker symbol (default: SPY)  
  - Strike price filters  
  - Y-axis choice: Strike or Moneyness  
- **Built with Streamlit** for a clean, interactive UI.  

## 🛠️ Tech Stack  
- **Streamlit** — Web app framework  
- **yfinance** — Financial data API  
- **NumPy / Pandas** — Data handling  
- **SciPy** — Implied volatility solver  
- **Plotly** — 3D visualization  

## 📝 License  
[MIT License](https://opensource.org/licenses/MIT) — Free to use, modify, and distribute.  

---  

Developed by [Anthony Le](https://www.linkedin.com/in/anthony-hn-le/)  

---

