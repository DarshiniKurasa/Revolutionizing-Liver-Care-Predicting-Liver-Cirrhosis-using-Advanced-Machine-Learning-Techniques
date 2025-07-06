from flask import Flask, render_template, request
import joblib
import pandas as pd 
import numpy as np
import traceback
import pickle

app = Flask(__name__)
# Load preprocessing objects and model

scaler = pickle.load(open("normalizer.pkl", "rb"))
model, feature_columns = pickle.load(open("logreg_liver_cirosis_model.pkl", "rb"))



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Form inputs
        Age = float(request.form["Age"])
        Quantity_of_alcohol_consumption = float(request.form["Quantity_of_alcohol_consumption"])
        Diabetes_Result = 1 if request.form["Diabetes_Result"].lower() == "yes" else 0
        BP = request.form["Blood_pressure"]
        Hemoglobin = float(request.form["Hemoglobin"])
        PCV = float(request.form["PCV"])
        Polymorphs = float(request.form["Polymorphs"])
        Lymphocytes = float(request.form["Lymphocytes"])
        Platelet_Count = float(request.form["Platelet_Count"])
        Indirect = float(request.form["Indirect"])
        Total_Protein = float(request.form["Total_Protein"])
        Albumin = float(request.form["Albumin"])
        Globulin = float(request.form["Globulin"])
        AG_Ratio = float(request.form["AG_Ratio"])
        AL_Phosphatase = float(request.form["AL_Phosphatase"])
        USG_Abdomen = 1 if request.form["USG_Abdomen"].lower() == "yes" else 0

        # 2. Compute derived feature
        systolic_pressure = float(BP.split('/')[0]) / float(BP.split('/')[1])

        # 3. Create base input dict
        input_dict = {
            "Age": Age,
            "Quantity_of_alcohol_consumption": Quantity_of_alcohol_consumption,
            "Diabetes_Result": Diabetes_Result,
            "Blood_pressure": systolic_pressure,
            "Hemoglobin": Hemoglobin,
            "PCV": PCV,
            "Polymorphs": Polymorphs,
            "Lymphocytes": Lymphocytes,
            "Platelet_Count": Platelet_Count,
            "Indirect": Indirect,
            "Total_Protein": Total_Protein,
            "Albumin": Albumin,
            "Globulin": Globulin,
            "AG_Ratio": AG_Ratio,
            "AL_Phosphatase": AL_Phosphatase,
            "USG_Abdomen": USG_Abdomen
        }

        # 4. Create a zero-filled DataFrame with all model columns
        x_input = pd.DataFrame(data=[np.zeros(len(feature_columns))], columns=feature_columns)

        # 5. Update only known fields
        for col in input_dict:
            if col in x_input.columns:
                x_input[col] = input_dict[col]

        # 6. Scale
        x_scaled = scaler.transform(x_input)

        # 7. Predict
        prediction = model.predict(x_scaled)[0]
        prediction_text = "Liver cirrhosis detected" if prediction == 1 else "No liver cirrhosis"

        return render_template("result.html", prediction_text=prediction_text)

    except Exception as e:
        error_message = f"Error during prediction: {e}"
        print(error_message)
        print(traceback.format_exc())
        return render_template("result.html", error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)




# model_training.ipynb (code to run in Jupyter Notebook)
# Save this in model_training.ipynb and run
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
# 1. Load and clean
data = pd.read_csv(r'C:\Users\ASUA\Downloads\liver.csv')
drop_cols = ['S.NO', 'Place(location where the patient lives)']
data.drop(columns=drop_cols, inplace=True, errors='ignore')
data.dropna(inplace=True)

# 2. Clean target column
target_col = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"
data = data[data[target_col].isin(['YES', 'NO'])]  # Filter for labeled samples
data[target_col] = data[target_col].map({'YES': 1, 'NO': 0})

# ✅ 3. One-hot encode *before* SMOTE
data = pd.get_dummies(data, drop_first=True)

# 4. Features & target
X = data.drop(columns=[target_col])
y = data[target_col]

# 5. Apply SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=2, random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Show balanced class distribution
print("After SMOTE:\n", y_resampled.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open("normalizer.pkl", "wb"))

# Train and evaluate models
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_model = None
best_score = 0

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))
    if acc > best_score:
        best_score = acc
        best_model = clf

# Save best model and feature names
pickle.dump((best_model, X.columns.tolist()), open("logreg_liver_cirosis_model.pkl", "wb"))
