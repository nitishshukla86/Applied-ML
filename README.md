# Applied-ML

### Objective: Make profit on stocks by buying at open price and selling at closing price by leveraging a binary classifier.
# Success metric
1. Profit earned in week   
2. Precision (Mminimum loss occured)
3.       

## ML Objective: Build a binary classifier to predict if closing price would be greater than the opening price after a predefined period of time. The input data is OHLC yfinance data.

# Input features: 
1. open-close delta history
2. Previous volume

# Target : max (0, sign(closing-opening)) 



## Benchmarks
1. Regress on closing-opening or both seperately
2. Say yes with probability p   

## Performance Metric
1. Accuracy
2. Precision
3. Increase F1 score