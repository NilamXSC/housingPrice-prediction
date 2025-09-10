🏡 Housing Prices ML Project

This project predicts house sale prices using the Kaggle House Prices dataset.  
It covers **EDA, feature engineering, model training, and deployment** as an interactive **Streamlit app**.

📂 Project Structure
data/ # training data (not included in repo)
models/ # trained models (.joblib files)
notebooks/
├── 01_exploration.ipynb
├── 02_modeling.ipynb
src/
└── train.py # script to retrain pipeline
app.py # Streamlit app
requirements.txt
README.md
.gitignore


🚀 How to run locally

Clone the repo:
   git clone https://github.com/YOUR_USERNAME/housing-prices.git
   cd housing-prices
Create and activate a virtual environment (Windows PowerShell):

powershell
python -m venv env
.\env\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

🌐 Deployment
The app is deployed on Streamlit Community Cloud.

✨ Features
📊 EDA built-in: summary, missing values, target distribution, correlations, boxplots, scatterplots

🤖 Machine Learning pipeline: preprocessing + RandomForestRegressor

🔮 Predictions: upload CSV or fill a quick form for single prediction

📈 Feature importances shown automatically

🎨 Styled Streamlit UI with tabs, spinners, and friendly instructions

📌 Notes
If your model file (final_model_tuned.joblib) is too large (>100 MB), GitHub/Streamlit Cloud may reject it.

Use src/train.py to generate a smaller demo model (e.g., n_estimators=50).

The training data (train.csv) is not included in this repo (Kaggle licensing).

👤 Author
Developed by Nilam Chakraborty