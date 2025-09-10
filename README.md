# 🏡 Housing Prices Prediction App

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://housingprice-prediction-38d67xeupwedq9cukkohph.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/NilamXSC/housingPrice-prediction)

A complete **end-to-end Data Science project** built on the Kaggle Housing Prices dataset.  
It covers **EDA, feature engineering, model training, and deployment** as an interactive **Streamlit web app**.

## ✨ Features

- 📊 **EDA built-in**  
  - Dataset summary (shape, missing values, statistics)  
  - Automatic visualizations: histograms, heatmaps, scatterplots, boxplots.

- 🤖 **Machine Learning Model**  
  - Preprocessing pipeline (imputation, scaling, one-hot encoding).
  - RandomForestRegressor trained on Kaggle’s Housing dataset.
  - Supports both **CSV upload** and **single record input**.

- 🔮 **Predictions**  
  - Upload CSV → get predictions + downloadable output.  
  - Enter details manually → instant prediction.

- 📈 **Feature Importances**  
  - Shows which features drive house prices.

- 🎨 **Streamlit App**  
  - Professional UI with tabs, spinners, and custom styling.
  - Friendly guidance (*“☕ Upload your CSV and sip your tea while we crunch the numbers…”*).


## 🚀 Live Demo

👉 [Try the app here](https://housingprice-prediction-38d67xeupwedq9cukkohph.streamlit.app)


## 🧑‍💻 Skills Learned / Technologies Used

- **Python for Data Science**
  - `pandas`, `numpy` → data wrangling & preprocessing  
  - `matplotlib`, `seaborn` → visualization  

- **Machine Learning**
  - `scikit-learn` pipelines  
  - Handling missing values, feature scaling, categorical encoding  
  - RandomForest modeling & evaluation (MAE, RMSE)  

- **Model Deployment**
  - `joblib` for model persistence  
  - Streamlit for web deployment  
  - Git & GitHub for version control  
  - Streamlit Community Cloud for hosting  

- **Software Engineering Practices**
  - Virtual environments (`venv`)  
  - Project structuring (`src/`, `models/`, `notebooks/`)  
  - `.gitignore` for clean repos  
  - Requirements management (`requirements.txt`)  


## 📂 Project Structure

housing-prices/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│ ├── final_model.joblib # demo model (committed for deployment)
│ └── final_model_tuned.joblib (optional)
├── data/
├── notebooks/
│ ├── 01_exploration.ipynb
│ └── 02_modeling.ipynb
└── src/
└── train_demo.py


## ⚡ How to Run Locally

1. Clone the repo:
   git clone https://github.com/NilamXSC/housingPrice-prediction
   cd housing-prices
   
2. Create and activate a virtual environment:
   python -m venv env
   # Windows
   .\env\Scripts\Activate.ps1
   
3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   streamlit run app.py

🙌 Acknowledgements
Dataset: Kaggle House Prices

Tools: Python, scikit-learn, Streamlit, GitHub

👤 Author
Developed by Nilam Chakraborty

GitHub: https://github.com/NilamXSC

LinkedIn: https://www.linkedin.com/in/chakrabortynilam9/
