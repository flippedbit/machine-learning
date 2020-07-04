import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def find_rmse(train_X, val_X, train_y, val_y, max_leaf = 2, previous_rmse = (0,0), prev_max_leaf = 0):
    rf_model = RandomForestRegressor(max_leaf_nodes = max_leaf, random_state=1)
    rf_model.fit(train_X, train_y)
    pred = rf_model.predict(val_X)
    rmse = mean_squared_error(val_y, pred)

    last_rmse, prev_rmse = previous_rmse
    if prev_max_leaf == 0:
        return find_rmse(train_X, val_X, train_y, val_y, max_leaf = max_leaf * 2, previous_rmse = (rmse, rmse), prev_max_leaf = max_leaf)

    #print("%d: %d (last: %d, prev: %d)" % (max_leaf, rmse, last_rmse, prev_rmse))
    if rmse > last_rmse:
        if rmse < prev_rmse:
            return find_rmse(train_X, val_X, train_y, val_y, max_leaf = int(prev_max_leaf * 0.5), previous_rmse=(rmse, prev_rmse), prev_max_leaf = max_leaf)
    else:
        return find_rmse(train_X, val_X, train_y, val_y, max_leaf = max_leaf * 2, previous_rmse = (rmse, last_rmse), prev_max_leaf = max_leaf)
    return (prev_max_leaf, last_rmse)
    

print("Opening data...")
file_path = "train.csv"
train_data = pd.read_csv(file_path)
y = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print("Finding optimal max_node...")
max, rmse = find_rmse(train_X, val_X, train_y, val_y)

print("Training data with %d nodes" % max)
rf_model = RandomForestRegressor(max_leaf_nodes = max, random_state = 1)
rf_model.fit(X, y)

print("Predicting values...")
file_path = "test.csv"
test_data = pd.read_csv(file_path)
X = test_data[features]
df = pd.DataFrame(data={"Id": test_data['Id'], "SalePrice": rf_model.predict(X)}, index=None)
print("Values:")
print(df.head())
print("Saving...")
#df.to_csv("prediction.csv", index=False)
