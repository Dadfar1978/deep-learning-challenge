
#Deep Learning Challenge – Alphabet Soup Charity Success Predictor
Bootcamp Module 21 | Binary Classification with Neural Networks

 
📌 Project Aim
Build a deeplearning model that predicts whether an applicant to Alphabet Soup, a fictional grantmaking foundation, will succeed (IS_SUCCESSFUL = 1) based on historical data for 34,000 funded organisations.
Accurate earlystage triage helps program officers focus their due diligence efforts and distribute funds more effectively.

 
🗂 Table of Contents
1. Quick Start
2. Dataset
3. Feature Engineering & Preprocessing
4. Model Architecture
5. Training & Evaluation
6. Results
7. Optimisation Experiments
8. Project Structure
9. Next Steps
10. Requirements
11. Author & Licence

 
Quick Start
# clone and enter repo
$ git clone https://github.com/Dadfar1978/deep-learning-challenge.git
$ cd deep-learning-challenge
 
# (optional) create virtual environment
$ conda create -n dlc python=3.11 tensorflowpandas scikit-learn matplotlib jupyterlab
$ conda activate dlc
 
# launch notebooks
$ jupyter lab AlphabetSoupCharit.ipynb

 
Dataset
File
Rows × Cols
Target
Notes
charity_data.csv
34,745 × 12
IS_SUCCESSFUL
Provided by Bootcamp; no missing values
Key categorical features:
• APPLICATION_TYPE (17 categories)
• CLASSIFICATION (71 categories)
Continuous / ordinal features:
• ASK_AMT – funding request amount
• INCOME – applicant revenue bracket
• STATUS – organisation age bucket

 
Preprocessing
1. Irrelevant columns removed: EIN, NAME
2. Rarelabel consolidation: categories with < 500 (APPLICATION_TYPE) or < 1 000 (CLASSIFICATION) records collapsed into Other
3. Onehot encoding: pd.get_dummies for all categorical variables
4. Train / test split: 75 / 25, random_state=78
5. Scaling: StandardScaler fit on training set only
Final input dimension: 116 features

 
Model Architecture 
Input (116)
│
├─ Dense(80, activation="relu")
│
├─ Dense(30, activation="relu")
│
└─ Dense(1, activation="sigmoid")  →  ŷ ∈[0,1]
• Loss: binary_crossentropy
• Optimizer: Adam (lr = 1e3)
• Epochs: 100
• Batch size: 32

 
Training & Evaluation
Split
Loss
Accuracy
Train
0.553
0.731
Test
0.558
0.724
Target accuracy (≥ 0.75) not yet achieved.

 
Results
• The baseline NN correctly classifies 72 % of heldout samples.
• Precision = 0.62, Recall = 0.11 ⇢ model skews toward the majority “notfunded” class.
• Confusion matrix and ROC curve plots available in AlphabetSoupCharit.ipynb.

 
Optimisation Experiments
Experiment
Change
Test Acc
Baseline
8030 units
0.724
+ Extra layer
804020
0.727
+ BatchNorm & Dropout(0.2)
804020
0.729
Class weights
class_weight={0:1,1:13}
0.742
Target encoding + class weights
embed / meanencode highcardinality features
0.762
Best run reaches 76.2 % – clearing the requirement.

 
Project Structure 
├─ AlphabetSoupCharit.ipynb           # baseline build & eval
├─ AlphabetSoupCharity_Optimization.ipynb  # tuning experiments
├─ Starter_Code.ipynb                # clean template provided by course
├─ charity_data.csv                  # dataset (gitignored in public repo)
├─ README.md                         # this file
└─ LICENSE                           # MIT

 
Next Steps
• 🎯 Hyperparameter search with Keras Tuner
• 🏷️ Embeddings for highcardinality categoricals (alternative to onehot)
• ⚖️ SMOTE or tf.data.experimental.rejection_resample to mitigate imbalance
• 📈 Compare against nonDL baselines (XGBoost, LightGBM, Logistic Reg.)
• 📢 SHAP explanations for feature importance + model transparency

 
Requirements 
python >= 3.10
tensorflow >= 2.15
pandas >= 2.2
scikitlearn >= 1.4
matplotlib >= 3.8
jupyterlab
Install via pip install -r requirements.txt (file included) or the conda snippet in Quick Start.

 
Author & Licence
Delaram Dadfarnia– Data Science Bootcamp Student
✉️ delaramrealtyca@gmail.com
🔗  • GitHub
Released under the MIT License – see LICENSE for details.
