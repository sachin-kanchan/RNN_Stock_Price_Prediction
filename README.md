# RNN Stock Price Prediction

## Project Overview
This project explores the application of Recurrent Neural Networks (RNNs) for predicting stock prices of major technology companies: Amazon (AMZN), Google (GOOGL), IBM, and Microsoft (MSFT). The objective is to develop models that can accurately forecast future stock prices based on historical data spanning from 2006 to 2018.

## Dataset
The dataset consists of daily stock market information for four tech companies with the following features:
- Date: Trading date (2006-2018)
- Open: Opening stock price
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing stock price
- Volume: Number of shares traded
- Name: Stock identifier

## Methodology
1. **Data Processing**: Aggregation of multi-company data, feature engineering, and time series preparation
2. **Exploratory Data Analysis**: Visualization of price trends, volume distributions, and correlation analysis
3. **Model Development**: Implementation and comparison of two RNN architectures:
   - Simple RNN model
   - LSTM (Long Short-Term Memory) model
4. **Hyperparameter Tuning**: Optimization of model configurations for improved performance
5. **Evaluation**: Assessment using multiple metrics (MSE, RMSE, MAE, MAPE, R²)

## Key Findings
- LSTM models significantly outperformed Simple RNN models for stock price prediction
- Optimal LSTM configuration: 1 LSTM layer, 128 units, 0.2 dropout rate, learning rate of 0.001, batch size of 16
- While showing promise, even the best model (LSTM) achieved an R² of only 0.2050, indicating room for improvement
- Trading volume distributions were right-skewed, with occasional significant spikes
- All stocks showed strong upward trends over time, with AMZN displaying particularly notable growth

## Future Work
- Incorporate technical indicators (Moving Averages, RSI, MACD, etc.)
- Integrate external data sources (macroeconomic factors, sentiment analysis)
- Experiment with advanced architectures (GRU, Attention mechanisms, Transformers)
- Implement ensemble methods combining multiple model types
- Develop multi-step forecasting capabilities

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow/Keras
- Scikit-learn

## Installation
```
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

## Usage
Run the Jupyter notebook `RNN_Stock_Price_Prediction.ipynb` alongside the necessary CSV files to reproduce the analysis and results.