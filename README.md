# Used-Ford-car-Price-Prediction
# Used Ford Car Price Prediction

A machine learning project that predicts the selling price of a used Ford car from its model, age, mileage, fuel type, transmission, tax, fuel economy and engine size, using **Linear Regression** in Python.

## Problem

What should a used car cost? Buyers and dealers want a fair price based on the car's details. I built a model that predicts the price and compared two ways of turning the car's text columns into numbers.

## Dataset

- File: `ford.csv`, 17,966 used Ford cars and 9 columns, with no missing values
- Columns: `model`, `year`, `price` (target), `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`
- Average price is about 12,280; the average mileage is about 23,400.


## Approach

1. **Exploratory data analysis** — price distribution, correlation heatmap, price against year, mileage, engine size, transmission, fuel type and model (box plots and scatter plots), and tax against mpg
2. **Encoding** — two methods compared:
   - **One-hot encoding** of `model`, `transmission` and `fuelType` (34 features in total)
   - **Label encoding** of the same three columns (8 features in total)
3. **Scaling** — standardised the numeric columns with `StandardScaler`
4. **Split** — 67% training and 33% testing (`random_state=42`), which gives 5,929 test cars
5. **Model** — `LinearRegression` from scikit-learn, trained once on each version of the data
6. **Evaluation** — R² and adjusted R² on the test set

## Results

| Encoding | R² (test) | Adjusted R² (test) |
| --- | --- | --- |
| One-hot encoding | 0.8397 | 0.8387 |
| Label encoding | 0.7310 | not calculated |

**One-hot encoding performed clearly better** (0.84 against 0.73). Label encoding gives each car model a number such as 5 or 13, which tells a linear model that the models have an order and that one is "bigger" than another. That is not true, and one-hot encoding avoids it.

## Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter Notebook

## Limitations and next steps

- The data has some unrealistic values (for example a car year of 2060 and engine sizes of 0) that I did not remove. Cleaning them is the next step.
- Linear Regression assumes straight-line relationships. Price usually falls in a curve with age and mileage, so tree-based models (Random Forest, Gradient Boosting) may do better.
- Add error metrics in price units (MAE and RMSE) and use cross-validation.
- Fit the scaler on the training data only, so no test information leaks into training.
- Strip extra spaces from the model names so that each model is only one category.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook ML_project3_.ipynb
```

## Author

Arpita Singh — [GitHub](https://github.com/Arpita-Singh16) · [LinkedIn](https://www.linkedin.com/in/arpita-singh-67b493339/)
