# HR_Analytics
Applying various machine learning techniques to identify the best algorithms to predict whether or not an employee will leave the company.

Dataset: HRdata.csv

Dataset obtained from Kaggle

## Instructions for package installation:

Installing Theano

pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

Installing Tensorflow

pip install tensorflow

Installing Keras

pip install --upgrade keras

## Description of Dataset:

1. satisfaction_level - satisfaction rating between 0-1
2. last_evaluation - last evaluation score between 0-1
3. number_project - number od projects the employee is currently on
4. average_montly_hours - average number of hours worked per month
5. time_spend_company - number of years working at the company
6. Work_accident - whether or not a worker had an accident during their time (binary)
7. promotion_last_5years - whether or not a worker got promoted in the last 5 years (binary)
8. dept - department the employee is working in: sales, accounting, hr, technical, support, management, IT, product_mng, marketing, RandD
9. salary - salary classified into 'low', 'medium', 'high'
10. left - whether or not the employee left the company (binary)

### Neural Network

Dummy variable: department
- technical: col 0
- support: col 1
- IT: col 0 & 2
- RandD: col 0 & 3
- accounting: col 0 & 4
- hr: col 0 & 5
- management: col 0 & 6
- marketing: col 0 & 7
- product_mng: col 0 & 8
- sales: col 0 & 9

Dummy variable - salary
- low: col 18 = 1
- medium: col 18 = 2
- high: col 18 = 0

## Machine Learning classification methods used:

- Logistic Regression: accuracy = 78.33%
- Decision Tree: accuracy = 96.97%
- Random Forest: accuracy = 98.36 (20 trees), 98.47% (100 trees)
- K-nearest neighbor: accuracy = 93.81% for k=5, 93.63 for k=10
- Naive Bayes: accuracy = 78.7%
- Support vector machines (linear, polynomial, radial, sigmoid kernels)
  - Linear: accuracy = 77.43%
  - Polynomial: accuracy = 94.87%
  - Radial basis: accuracy = 94.90%
  - Sigmoid: accuracy = 57.20%
