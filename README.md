# 📈 Mean-Variance Portfolio Optimization by CNN-GRU-LSTM Based Predictions

This project combines deep learning-based stock price prediction with classical financial theory to build an optimized investment portfolio. By integrating **CNN, GRU, and LSTM models** for time series forecasting with **Markowitz Mean-Variance Optimization**, it delivers an effective strategy for maximizing returns while minimizing risk.

The portfolio optimization process is powered by **Monte Carlo Simulation**, generating thousands of potential portfolios and selecting the optimal ones based on Sharpe ratio and volatility.

---

## 🔍 Project Overview

- Forecasting stock prices using hybrid deep learning models (CNN + GRU + LSTM).
- Applying Mean-Variance Portfolio Optimization to allocate weights efficiently.
- Using **Monte Carlo Simulation** to evaluate over 100,000 random portfolios.
- Identifying portfolios with **maximum Sharpe ratio** and **minimum variance**.
- Comparing optimized portfolio performance against NASDAQ benchmark.

---

## ✅ Predictive Model Performance

The model achieved strong predictive accuracy based on R-squared values:

- AAPL: 0.971  
- TSLA: 0.953  
- META: 0.982  
- GOOGLE: 0.962  
- NVIDIA: 0.992  
- AVGO: 0.985  
- MSFT: 0.978

---

## 📈 Portfolio Optimization Results

- Max Sharpe ratio cumulative return: **78.81%**  
- Minimum variance cumulative return: **60.25%**  
- NASDAQ benchmark cumulative return: **29.43%**

These results highlight the power of combining AI-driven forecasting with traditional financial optimization techniques.

---

## 💡 Monte Carlo Simulation Technique

A Monte Carlo simulation approach was used to explore a wide space of portfolio combinations and identify the most efficient portfolios. The simulation calculates portfolio returns, volatility, and Sharpe ratio for each iteration and selects the optimal configurations.

Sample methodology:

```python
def calculate_returns(forecast_results):
    forecast_df = pd.DataFrame(forecast_results)
    log_returns = np.log(forecast_df / forecast_df.shift(1)).dropna()
    return log_returns

def simulate_portfolios(log_returns, n_simulations=100000, risk_free_rate=0.02/252):
    n_assets = log_returns.shape[1]
    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': []}
    
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    for _ in range(n_simulations):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        
        results['Returns'].append(port_return)
        results['Volatility'].append(port_volatility)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)
    
    return pd.DataFrame(results), mean_returns, cov_matrix

def get_optimal_portfolios(portfolios_df):
    max_sharpe_idx = portfolios_df['Sharpe'].idxmax()
    min_vol_idx = portfolios_df['Volatility'].idxmin()
    
    max_sharpe_portfolio = portfolios_df.loc[max_sharpe_idx]
    min_variance_portfolio = portfolios_df.loc[min_vol_idx]
    
    return max_sharpe_portfolio, min_variance_portfolio

