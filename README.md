# ChurnSense — Customer Churn Prediction

A machine learning project that predicts customer churn using the Telco Customer Churn dataset. It includes an interactive Streamlit dashboard for viewing churn predictions and estimated churn probabilities.

## Features

- Interactive Streamlit dashboard
- Predicts **Churn** or **No Churn**
- Displays the estimated churn probability
- Preprocessing and prediction through a saved machine learning pipeline
- Jupyter Notebook for model development and evaluation

## Model Performance

The Logistic Regression model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1-score | 60.40% |

These scores describe the model's test-set performance and do not guarantee predictions for individual customers.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_pipeline.joblib
└── notebooks/
    └── your-notebook.ipynb
```

> The dataset and notebook filenames above are examples. Update them if your actual filenames differ.

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/RahulMalhi10/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit dashboard:

   ```bash
   streamlit run app.py
   ```

## Model File

The dashboard loads the trained pipeline from `models/churn_pipeline.joblib`. The scikit-learn version in `requirements.txt` should match the version used to save the model.

## Dataset

This project uses the Telco Customer Churn dataset. Check the dataset's original source and license before redistributing the CSV.

## Deployment

To deploy on Streamlit Community Cloud, connect this GitHub repository, select `app.py` as the main file, and ensure `requirements.txt` and `models/churn_pipeline.joblib` are committed to the repository.

## Author

**Rahul Malhi**  
BS Computer Science Student | Aspiring AI/ML Engineer

Interested in Machine Learning, Data Science, and building practical AI applications.

- GitHub: [@RahulMalhi10](https://github.com/RahulMalhi10)

---

If you improve this project, consider adding a screenshot of the dashboard and a link to the deployed live demo.
