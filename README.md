# ValueInvestor

## **Background**
ValueInvestor is an intelligent system designed to aid portfolio investment decisions in emerging markets using stock market data. Our goal is to maximize capital returns while minimizing losses through predictive modeling and strategic stock recommendations based on **value investing principles**. We focus on intrinsic value rather than daily market volatility, making investment decisions on a weekly, monthly, and quarterly basis.

## **Data Description**
The dataset consists of **portfolio companies' trading data** from emerging markets, spanning:
- **2020 Q1-Q4 stock prices** (used for training models)
- **2021 Q1 stock prices** (used for evaluation)
- Each company’s stock data is stored in different sheets
- Market operating days vary based on country and stock exchange

**Data Source:** [Stock Trading Data](https://docs.google.com/spreadsheets/d/1MiunF_O8eNWIcfaOA4PVm668RN7FgLNA0a6U4LWf5Bk/edit?usp=sharing)

## **Goals**
- **Predict stock price valuations** on a **daily, weekly, and monthly** basis.
- **Generate BUY, HOLD, or SELL recommendations** to maximize profitability.
- **Minimize capital losses** and **reduce hold period**.
- **Evaluate strategy effectiveness** using Bollinger Bands.

---

## **Models Implemented**
To achieve our investment goals, we implemented **five time series forecasting models**:

### 1. **ARMA (AutoRegressive Moving Average)**
- Used for analyzing and predicting future stock price trends.
- Assumes stock prices follow a stationary time series.
- Decomposed price data into **trend, seasonality, and residuals** before applying ARMA.

### 2. **ARIMAX (AutoRegressive Integrated Moving Average with Exogenous Variables)**
- An extension of ARIMA that incorporates **exogenous variables (e.g., volume, market trends, other stock indicators)**.
- Trained the model on **2020 stock data** and evaluated its effectiveness on **2021 Q1 stock data**.
- Tuned hyperparameters **(p, d, q)** to optimize predictions.

### 3. **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
- Enhances ARIMA by accounting for **seasonality** in stock prices.
- Modeled using seasonal components **(P, D, Q, S)**.
- Evaluated effectiveness using **Mean Squared Error (MSE) and Mean Absolute Error (MAE)**.

### 4. **Prophet Model (by Facebook)**
- Designed for handling time series data with strong trends and seasonality.
- Applied Prophet to forecast future stock prices and generate decision recommendations.
- Evaluated accuracy using **MSE and MAE metrics**.

### 5. **LSTM (Long Short-Term Memory) Neural Network**
- Implemented a deep learning model to **capture long-term dependencies in stock trends**.
- Preprocessed stock price data and trained an **LSTM-based predictive model**.
- Generated **BUY, HOLD, and SELL decisions** based on forecasted trends.

---

## **Decision Strategy**
To determine investment actions, we applied **Bollinger Bands**:
- **BUY**: When the forecasted price **falls below the Lower Band**.
- **SELL**: When the forecasted price **rises above the Upper Band**.
- **HOLD**: When the forecasted price **remains between the bands**.

### **Final Decision Implementation**
After predicting stock prices with each model, we created a **decision function** to categorize recommendations:
```python
# Generate Decision based on Bollinger Bands
forecast['Decision'] = 'Hold'
for i in range(len(forecast)):
    if forecast['rolling_mean_diff'].iloc[i] < forecast['Lower_Band'].iloc[i]:
        forecast['Decision'].iloc[i] = 'Buy'
    elif forecast['rolling_mean_diff'].iloc[i] > forecast['Upper_Band'].iloc[i]:
        forecast['Decision'].iloc[i] = 'Sell'
    else:
        forecast['Decision'].iloc[i] = 'Hold'
```

---

## **Results & Evaluation**
- **MSE (Mean Squared Error)** and **MAE (Mean Absolute Error)** were used to evaluate model performance.
- **LSTM and Prophet models** performed best for long-term forecasting.
- **ARIMAX provided the most accurate short-term predictions**.
- **Bollinger Bands combined with forecasting models** led to an optimal trading strategy.

### **Visualization**
- **Time series predictions** were plotted against **actual stock prices**.
- **Bollinger Bands** were overlaid to assess decision accuracy.
- **BUY and SELL signals** were highlighted for investment recommendations.

---

## **Conclusion**
The **ValueInvestor** system successfully combines **traditional statistical models (ARMA, ARIMAX, SARIMA)** and **modern deep learning models (LSTM, Prophet)** to create an intelligent stock investment decision-making tool. By leveraging time-series forecasting, exogenous variables, and Bollinger Bands, this approach optimizes stock trading strategies to **maximize capital gains and minimize risks**. After comprehensive analysis, ARIMAX emerged as the most effective model, consistently delivering superior capital returns while minimizing losses. This outcome highlights the value of incorporating exogenous variables into forecasting models, as ARIMAX's enhanced predictive power made it the most reliable tool for optimizing investment decisions in emerging markets.

