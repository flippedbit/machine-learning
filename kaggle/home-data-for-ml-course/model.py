import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def get_score(X_train, y_train, n_estimators = 10, learning_rate = 0.1):
    from sklearn.model_selection import cross_val_score
    my_pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('model', XGBRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate)),
                        ])
    scores = -1 * (cross_val_score(my_pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))
    return scores.mean()

print("Opening data...")
file_path = "train.csv"
train_data = pd.read_csv(file_path)

print("Pre-process data")
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

train_X_full, val_X_full, train_y, val_y = train_test_split(train_data, y, random_state=0)

low_cardinality_cols = [col for col in train_X_full.columns if train_X_full[col].nunique() < 10 and train_X_full[col].dtype == "object"]
num_cols = [col for col in train_X_full.columns if train_X_full[col].dtype in ['int64', 'float64']]
cols = low_cardinality_cols + num_cols

train_X = train_X_full[cols].copy()
val_X = val_X_full[cols].copy()

numerical_transofrmer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehost', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transofrmer, num_cols),
        ('cat', categorical_transformer, low_cardinality_cols)
    ]
)

print("Optimize model parameters")
estimators_results = {}
for num in range(50, 1000, 50):
    estimators_results[num] = get_score(X_train=train_X, y_train=train_y, n_estimators=num)
n_estimators=min(estimators_results, key = lambda i: estimators_results[i])
print(" - using %d n_estimators in model (MAE: %.5f)" % (n_estimators, estimators_results[n_estimators]))

learning_results = {}
from numpy import arange
for num in arange(0.01, 0.2, 0.01):
    learning_results[num] = get_score(X_train = train_X, y_train = train_y, n_estimators = n_estimators, learning_rate = num)
learning_rate = min(learning_results, key = lambda i: learning_results[i])
print(" - using %.2f learning_rate in model (MAE: %.5f)" % (learning_rate, learning_results[learning_rate]))

model = XGBRegressor(random_state = 0, n_estimators = n_estimators, learning_rate = learning_rate)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(train_X, train_y)

print("Predicting values...")
file_path = "test.csv"
test_data = pd.read_csv(file_path)
X = test_data[cols].copy()
df = pd.DataFrame(data={"Id": test_data['Id'], "SalePrice": pipeline.predict(X)}, index=None)
print("Values:")
print(df.head())
print("Saving...")
df.to_csv("gboosted_prediction.csv", index=False)