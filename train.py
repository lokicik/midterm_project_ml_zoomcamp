import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,f1_score,classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

df = pd.read_csv("data.csv")
df_original = pd.read_csv("data.csv")

possible_na = ["waistline", "sight_left", "sight_right", "SGOT_AST", "gamma_GTP"]

def grab_col_names(df, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numeric, and categorical cardinal variables in the dataset.
    Note: Numeric variables with a categorical appearance are also included in the categorical variables.

    Parameters
    ------
        df: DataFrame
                The DataFrame for which variable names are to be retrieved.
        cat_th: int, optional
                Class threshold value for numeric but categorical variables.
        car_th: int, optional
                Class threshold value for categorical but cardinal variables.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numeric variables.
        cat_but_car: list
                List of categorical variables that appear to be cardinal.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = the total number of variables.
        num_but_cat is included in cat_cols.
        The sum of the 3 lists returned is equal to the total number of variables: cat_cols + num_cols + cat_but_car = the number of variables.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations(Rows): {df.shape[0]}")
    print(f"Variables(Columns): {df.shape[1]}\n")
    print(
        f'cat_cols: {len(cat_cols)}\nnum_cols: {len(num_cols)}\ncat_but_car: {len(cat_but_car)}\nnum_but_cat: {len(num_but_cat)}')
    print(f"\ncat_cols: {cat_cols}\nnum_cols: {num_cols}\ncat_but_car: {cat_but_car}\n")

    print(
        f"\ncat_cols data types:\n\n{df[cat_cols].dtypes}\n\nnum_cols data types:\n\n{df[num_cols].dtypes}\n\ncat_but_car data types:\n\n{df[cat_but_car].dtypes}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["DRK_YN"] = df["DRK_YN"].map({"Y": 1, "N": 0})

smk_stat_mapping = {1: 0, 2: 1, 3: 2}

# Apply the mapping to the column
df['SMK_stat_type_cd'] = df['SMK_stat_type_cd'].map(smk_stat_mapping)

print(possible_na)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


outlier_cols = []
for col in num_cols:
    if check_outlier(df, col):
        outlier_cols.append(col)
        print(col)


def winsorize(dataframe, col_name, lower_quantile=0.05, upper_quantile=0.95):
    lower_limit, upper_limit = outlier_thresholds(dataframe, col_name, q1=lower_quantile, q3=upper_quantile)

    # Apply Winsorization
    dataframe[col_name] = dataframe[col_name].apply(
        lambda x: lower_limit if x < lower_limit else (upper_limit if x > upper_limit else x))


# Defining your lower and upper quantiles for Winsorization
lower_quantile = 0.05
upper_quantile = 0.95

# Apply Winsorization
for col in outlier_cols:
    winsorize(df, col, lower_quantile, upper_quantile)

for column in possible_na:
    df = df[df[column] != df[column].max()]
df = df.drop_duplicates()

# BMI = weight(kg) / (height(m)^2)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

conditions = [
    (df['BMI'] < 18.5),
    (df['BMI'] >= 18.5) & (df['BMI'] < 25),
    (df['BMI'] >= 25) & (df['BMI'] < 30),
    (df['BMI'] >= 30)
]

# labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']

df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, float('inf')], labels=['0', '1', '2', '3'])

# Mean arterial pressure (MAP) = DBP + (SBP - DBP) / 3
df['MAP'] = df['DBP'] + (df['SBP'] - df['DBP']) / 3

# Liver_Enzyme_Ratio = SGOT_AST / SGOT_ALT
df['Liver_Enzyme_Ratio'] = df['SGOT_AST'] / df['SGOT_ALT']

# Anemia_Indicator if hemoglobin < 12 --> anemia
anemia_threshold = 12
df['Anemia_Indicator'] = (df['hemoglobin'] < anemia_threshold).astype(int)

from sklearn.preprocessing import MinMaxScaler

smaller_dataset = df.sample(n=20000, random_state=42)

columns_to_scale = df.columns.difference(["DRK_YN", "SMK_stat_type_cd"])

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df[columns_to_scale])
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

scaled_dataset = df.copy()
scaled_dataset = scaled_dataset.reset_index(drop=True)  # Reset the index
scaled_dataset[columns_to_scale] = scaled_df

##########################################################################################
##################################### TRAINING DRK_YN ####################################
##########################################################################################
X = scaled_dataset.drop(["DRK_YN", "SMK_stat_type_cd"], axis=1)
y = scaled_dataset["DRK_YN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

results["LGBM"] = {'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 17}
results["XGB"] = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 123}
results["RF"] = {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 199}

from sklearn.ensemble import VotingClassifier

best_params_lgbm = results["LGBM"]
best_params_xgb = results["XGB"]
best_params_rf = results["RF"]

lgbm = LGBMClassifier(**best_params_lgbm)
xgb = XGBClassifier(**best_params_xgb)
rf = RandomForestClassifier(**best_params_rf)

drinking_model = VotingClassifier(estimators=[
    ("LGBM", lgbm),
    ("XGB", xgb),
    ("RF", rf)
], voting="soft")  # You can use "hard" or "soft" voting

drinking_model.fit(X_train, y_train)

drinking_predictions = drinking_model.predict(X_test)

accuracy = accuracy_score(y_test, drinking_predictions)

precision = precision_score(y_test, drinking_predictions)

recall = recall_score(y_test, drinking_predictions)

f1 = f1_score(y_test, drinking_predictions)

roc_auc = roc_auc_score(y_test, drinking_predictions)

print("Drinking Prediction Model Performance")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

#######################################################################################
######################### TRAINING SMK_stat_type_cd ###################################
#######################################################################################
X = scaled_dataset.drop(["DRK_YN", "SMK_stat_type_cd"], axis=1)
y = scaled_dataset["SMK_stat_type_cd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results["LGBM"] = {'learning_rate': 0.05, 'max_depth': 9, 'num_leaves': 16}
results["XGB"] = {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 182}
results["RF"] = {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 182}

from sklearn.preprocessing import label_binarize

best_params_lgbm = results["LGBM"]
best_params_xgb = results["XGB"]
best_params_rf = results["RF"]

lgbm = LGBMClassifier(**best_params_lgbm)
xgb = XGBClassifier(**best_params_xgb)
rf = RandomForestClassifier(**best_params_rf)

smoking_model = VotingClassifier(estimators=[
    ("LGBM", lgbm),
    ("XGB", xgb),
    ("RF", rf)
], voting="soft")

smoking_model.fit(X_train, y_train)

smoking_predictions = smoking_model.predict(X_test)
smoking_predictions_v2 = smoking_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, smoking_predictions)

precision = precision_score(y_test, smoking_predictions, average='weighted')

recall = recall_score(y_test, smoking_predictions, average='weighted')

f1 = f1_score(y_test, smoking_predictions, average='weighted')

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(y_test_bin, smoking_predictions_v2, average='macro')

print("Smoking Prediction Model Performance")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

expected_feature_names = [
    'sex', 'age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right',
    'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole',
    'LDL_chole', 'triglyceride', 'hemoglobin', 'urine_protein', 'serum_creatinine',
    'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'BMI', 'BMI_Category', 'MAP', 'Liver_Enzyme_Ratio', 'Anemia_Indicator'
]

my_record = X_test.sample(n=1)[expected_feature_names]

import json

json_data = my_record.to_json(orient='records', lines=True)
parsed_data = json.loads(json_data)
pretty_json = json.dumps(parsed_data, indent=4)
print(f"For input:\n{pretty_json}\n")

drinking_prediction = drinking_model.predict(my_record)
predicted_class_drinking = drinking_prediction[0]
print(f"DRK_YN (1 for Yes / 0 for No):\nPrediction: {predicted_class_drinking}\n")

smoking_prediction = smoking_model.predict(my_record)
predicted_class_smoking = smoking_model.classes_[smoking_prediction[0]]
print(
    f"SMK_stat_type_cd (Smoking Status 1 for Never Smoked) / 2 for Used to Smoke / 3 for Still Smoking)):\nPrediction: {predicted_class_smoking}")

# Save the model
import pickle

output_file = f'drinking_model.bin'
output_file

with open(output_file, 'wb') as f_out:
    pickle.dump((drinking_model), f_out)

# Save the model
import pickle

output_file = f'smoking_model.bin'
output_file

with open(output_file, 'wb') as f_out:
    pickle.dump((smoking_model), f_out)



