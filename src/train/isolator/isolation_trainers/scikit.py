from isolation_trainer import IsolationTrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

class ScikitIsolationTrainer(IsolationTrainer):

    def __init__(self, regressor=GradientBoostingRegressor(random_state=0)): 
        self.regressor = regressor
        self.scaler = None
        self.model = None

    def train(self, data, feature_cols, label_col, test_size=0.1):
        y_values = data[label_col].values
        x_values = data[feature_cols].values
        self.scaler = MinMaxScaler()
        features = self.scaler.fit_transform(x_values)
        # split train,test  
        X_train, _, y_train, _ = train_test_split(features, y_values, test_size=test_size, shuffle=True)
        model = model.fit(X_train, y_train)

    def predict(self, data, feature_cols):
        x_values = data[feature_cols].values
        features = self.scaler.transform(x_values)
        predicted_power = self.model.predict(features).squeeze()
        return predicted_power

