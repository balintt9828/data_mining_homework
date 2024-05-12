import pandas as pd
import numpy as np


class Node:
    def __init__(
            self,
            min_samples_split: object = None,
            min_samples_leaf: object = None,
            max_depth: object = None,
            depth: object = None,
            node_type: object = None,
            rule: object = None,
            method: object = None
    ) -> object:

        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.min_samples_leaf = min_samples_leaf if min_samples_leaf else 1
        self.max_depth = max_depth if max_depth else 5

        self.depth = depth if depth else 0
        self.node_type = node_type if node_type else "root"
        self.rule = rule if rule else ""
        self.method = method if method else "MSE"

        self.left = None
        self.right = None
        self.features = None
        self.best_feature = None
        self.best_value = None

        self.impurity = 0
        self.len = 0
        self.y_mean = 0
        self.y_hat = 0

    @staticmethod
    def __calculate_mse(y_true, y_hat) -> float:

        n = len(y_true)
        r = y_true - y_hat
        r = r ** 2
        r = np.sum(r)
        mse = r / n

        return mse

    @staticmethod
    def __calculate_mae(y_true, y_hat) -> float:
        n = len(y_true)
        r = y_true - y_hat
        r = np.abs(r)
        r = np.sum(r)
        mae = 1 / n * r

        return mae

    @staticmethod
    def __ma(x: np.array, window: int) -> np.array:
        return np.convolve(x, np.ones(window), 'valid') / window

    def __calculate_impurity(self, y_true, y_hat) -> float:
        if self.method == "MAE":
            return self.__calculate_mae(y_true, y_hat)
        else:
            return self.__calculate_mse(y_true, y_hat)

    @staticmethod
    def split_data(df, random_state=None, ratio=None):
        if random_state is None:
            random_state = np.random.random.randint(1, high=100, size=1)
        if ratio is None:
            ratio = 0.2

        np.random.seed(random_state)
        indexes = [i for i in range(len(df))]
        np.random.shuffle(indexes)

        test_indexes = indexes[:round(len(df) * ratio)]
        test_df = pd.DataFrame()
        train_df = pd.DataFrame()

        for number in range(len(df)-1, 0, -1):
            if number in test_indexes:
                test_df = pd.concat([test_df, pd.DataFrame([df.iloc[number]])], ignore_index=True)
            else:
                train_df = pd.concat([train_df, pd.DataFrame([df.iloc[number]])], ignore_index=True)

        return train_df, test_df

    def fit(self, X, Y):
        self.len = len(X)
        self.y_mean = np.mean(Y)
        self.impurity = self.__calculate_impurity(Y, self.y_mean)
        self.features = list(X.columns)
        self.y_hat = np.bincount(Y).argmax()

        df = X.copy()
        df['Y'] = Y

        if (self.depth < self.max_depth) and (len(Y) >= self.min_samples_split):
            best_feature, best_value = self.__best_split(X, Y)

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value
                left_df, right_df = df[df[best_feature] <= best_value].copy(), df[df[best_feature] > best_value].copy()

                self.left = self.__create_node(best_feature, best_value, 'left_node')
                self.left.fit(X=left_df[list(X.columns)], Y=left_df['Y'].values.tolist())

                self.right = self.__create_node(best_feature, best_value, 'right_node')
                self.right.fit(X=right_df[list(X.columns)], Y=right_df['Y'].values.tolist())

    def print(self):
        self.__print_info()

        if self.left is not None:
            self.left.print()

        if self.right is not None:
            self.right.print()

    def predict(self, X: pd.DataFrame):
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})

            predictions.append(self.__predict_obs(values))

        return predictions

    def __best_split(self, X, Y) -> tuple:
        df = X.copy()
        df['Y'] = Y
        impurity_base = self.impurity
        best_feature = None
        best_value = None

        for feature in list(X.columns):
            x_df = df.dropna().sort_values(feature)
            x_means = self.__ma(x_df[feature].unique(), 2)

            for value in x_means:
                left_y = x_df[x_df[feature] < value]['Y'].values
                right_y = x_df[x_df[feature] >= value]['Y'].values
                impurity_split = self.__create_impurity_of_split(left_y, right_y)

                if self.__is_best_cut(impurity_base, impurity_split, left_y, right_y):
                    best_feature = feature
                    best_value = value
                    impurity_base = impurity_split

        return best_feature, best_value

    def __is_best_cut(self, impurity_base, impurity_split, left_y, right_y):
        impurity_criteria = impurity_split < impurity_base
        samples_len_crit_left = len(left_y) >= self.min_samples_leaf
        samples_len_crit_right = len(right_y) >= self.min_samples_leaf
        best_cut = impurity_criteria and samples_len_crit_left and samples_len_crit_right
        return best_cut

    def __create_impurity_of_split(self, left_y, right_y):
        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)

        res_left = left_y - left_mean
        res_right = right_y - right_mean

        res = np.concatenate((res_left, res_right))
        res_mean = np.mean(res)

        impurity_split = self.__calculate_impurity(res, res_mean)

        return impurity_split

    def __create_node(self, best_feature, best_value, node_type):
        return Node(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            node_type=node_type,
            rule=f"{best_feature} <= {round(best_value, 3)}",
            method=self.method
        )

    def __print_info(self, width=4):
        indentation = int(self.depth * width)
        spaces = "─" * (width-1)

        if self.node_type == 'root':
            print(f"{' ' * indentation}└{spaces}┬ Root")
        else:
            print(f"{' ' * indentation}└{spaces}┬ Split rule: {self.rule}")
        print(f"{' ' * indentation}{' ' * width}├ {self.method} of the node: {round(self.impurity, 2)}")
        print(f"{' ' * indentation}{' ' * width}├ Count of observations in node: {self.len}")
        print(f"{' ' * indentation}{' ' * width}└ Class of node: {self.y_hat}")

    def __predict_obs(self, values: dict) -> int:
        cur_node = self

        while cur_node.right is not None:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if values.get(best_feature) < best_value:
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right

        return cur_node.y_hat
