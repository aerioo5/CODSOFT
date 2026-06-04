# Sales Prediction Using Python – Analysis

## Project Overview

This project predicts product sales based on advertising expenditures across different media channels:

- TV Advertising
- Radio Advertising
- Newspaper Advertising

The goal is to understand how marketing investments influence sales and build a machine learning model capable of forecasting future sales.

## Dataset Summary

Your uploaded dataset contains:

| Feature   | Description                            |
| --------- | -------------------------------------- |
| TV        | Advertising budget spent on TV         |
| Radio     | Advertising budget spent on Radio      |
| Newspaper | Advertising budget spent on Newspapers |
| Sales     | Product sales (Target Variable)        |

Dataset Size: 200 records × 4 columns

## Statistical Overview

| Metric  | TV     | Radio | Newspaper | Sales |
| ------- | ------ | ----- | --------- | ----- |
| Mean    | 147.04 | 23.26 | 30.55     | 15.13 |
| Maximum | 296.4  | 49.6  | 114.0     | 27.0  |
| Minimum | 0.7    | 0.0   | 0.3       | 1.6   |

## Model Used

- Linear Regression
- Train-Test Split: 80% Training, 20% Testing

## Results

After training the model:

- R² Score: 0.906
- Mean Absolute Error (MAE): 1.27

## Interpretation

- The model explains approximately 90.6% of the variation in sales, indicating excellent predictive performance.

- On average, predictions differ from actual sales by only 1.27 units.

## Feature Impact

Model coefficients:

| Feature   | Impact on Sales |
| --------- | --------------- |
| TV        | +0.0545         |
| Radio     | +0.1009         |
| Newspaper | +0.0043         |

## Observation:

- Radio advertising has the strongest influence on sales.

- TV advertising also contributes significantly.

- Newspaper advertising has very little impact compared to TV and Radio.

## Key Insights

1. Sales increase as TV and Radio advertising budgets increase.

2. Radio advertisements show the highest effectiveness.

3. Newspaper advertisements contribute minimally to sales growth.

4. Linear Regression performs very well for this dataset.

## Conclusion

The Sales Prediction model successfully forecasts product sales using advertising data. With an R² score of 90.6%, the model demonstrates strong predictive capability and can help businesses optimize advertising budgets and improve marketing strategies.

## Future Improvements

- Try Random Forest Regression and XGBoost.

- Perform feature engineering.

- Add seasonal and customer demographic data.

- Deploy the model as a web application using Flask or Streamlit.
