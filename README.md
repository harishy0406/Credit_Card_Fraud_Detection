## Credit Card Fraud Detection (Machine Learning)

**Aim**  
Detect fraudulent credit card transactions using advanced machine learning techniques.

**Description**  
This project applies and compares multiple classification algorithms to identify potentially fraudulent transactions in an imbalanced dataset. It includes:
- Data loading and exploratory analysis
- Handling of class imbalance (e.g., class weights, resampling)
- Training and evaluation of different models (Logistic Regression, Random Forest, Gradient Boosting, etc.)
- Metrics focused on fraud detection (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix)

**Tech Stack**
- Python
- Pandas, NumPy
- Scikit-learn
- (Optional) Imbalanced-learn, Matplotlib/Seaborn

**Project structure**
- `data/` – place your raw dataset here (e.g., `creditcard.csv`)
- `notebooks/` – exploratory analysis and experiments
- `src/` – reusable code (data processing, models, evaluation)
- `requirements.txt` – Python dependencies

**How to run**
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put your dataset into `data/` (e.g., `data/creditcard.csv`).
3. Run the example end-to-end script:
   ```bash
   python src/train_baseline.py
   ```
   
**Snapshots**


