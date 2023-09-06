from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.base import BaseEstimator, ClassifierMixin

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lstm_units=50, dropout_rate=0.2, optimizer='adam', epochs=10, batch_size=32, verbose=0):
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
        model.add(LSTM(self.lstm_units, dropout=self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def score(self, X, y):
        _, accuracy = self.model.evaluate(X, y, verbose=self.verbose)
        return accuracy