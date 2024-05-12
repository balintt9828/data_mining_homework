import pandas as pd
from DecisionTree import Node

wine_data = pd.read_csv('./datasets/winequality-red.csv')

model = Node(method="MSE")
train_split, test_split = model.split_data(wine_data, random_state=42, ratio=0.01)

X_train = train_split.iloc[:, :-1]
Y_train = train_split[['quality']]
model.fit(X=X_train, Y=Y_train['quality'].values.tolist())
model.print()

X_test = test_split.iloc[:, :-1]
X_test['yhat'] = model.predict(X_test)
print(X_test)


