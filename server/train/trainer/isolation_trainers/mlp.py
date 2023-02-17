from isolation_trainer import IsolationTrainer
import keras
from keras import layers
from tensorflow.keras.constraints import non_neg as nonneg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def mlp_1(input_dim):
    model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
    return model

def mlp_2(input_dim):
    model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(250, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
    return model

def mlp_3(input_dim):
    model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(250, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(150, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
    return model

def mlp_2_do(input_dim):
    model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dropout(0.5),
    layers.Dense(250, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dropout(0.25),
    layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
    return model

def mlp_3_do(input_dim):
    model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dropout(0.5),
    layers.Dense(250, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dropout(0.25),
    layers.Dense(150, activation='relu', input_shape=[input_dim], kernel_constraint=nonneg()),
    layers.Dropout(0.125),
    layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
    return model

class MLPIsolationTrainer(IsolationTrainer):

    def __init__(self, build_model_function=mlp_1): 
        self.build_model_function = build_model_function
        self.scaler = None
        self.model = None

    def train(self, data, feature_cols, label_col, test_size=0.1):
        y_values = data[label_col].values
        x_values = data[feature_cols].values
        self.scaler = MinMaxScaler()
        features = self.scaler.fit_transform(x_values)
        # split train,test  
        X_train, X_test, y_train, y_test = train_test_split(features, y_values, test_size=test_size, shuffle=True)
        X_train = np.asarray(X_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        input_dim = features.shape[1]
        self.model = self.build_model_function(input_dim)

        self.model.fit(X_train, y_train,
                    epochs=1000,
                    verbose=0,
                    validation_data=(X_test, y_test))

    def predict(self, data, feature_cols):
        x_values = data[feature_cols].values
        features = self.scaler.transform(x_values)
        predicted_power = self.model.predict(features).squeeze()
        return predicted_power

