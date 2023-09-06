import pandas as pd
import numpy as np
import re, random
from gensim.models import Word2Vec

from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from clearWords import ClearWords
from lstm_classifier import LSTMClassifier

def vectorizeText(model, x):
    return np.mean([model.wv[word] for word in x if word in model.wv], axis=0)

#########################
### Leitura das Bases ###
#########################
df = pd.read_csv("churn_com_texto.csv", dtype=str, on_bad_lines='skip')

################
### Pré Proc ###
################
df['Comentários'] = df['Comentários'].fillna(df['Número de Reclamações'])
df['Número de Reclamações'] = df['Número de Reclamações'].apply(lambda x: re.sub('^.*(?!(^(\d|-)$)).$|2', '1', x)).str.replace('-', '0')
df['Volume de Dados'] = df['Volume de Dados'].fillna('-').apply(lambda x: re.sub('(?i)^(?!(^\d.+gb$)).*$', '-', x))

cl = ClearWords(df)
df['Comentário Tratado'] = df['Comentários']

metodos = ['lowerCaseWords', 'removeStopWords', 'removeSlang', 'removerAcentos', 'removeStopWords', 'removeEquals', 'removerLetrasDuplicadas', 'removeCharacters', 'createRads']
df['Comentário Tratado'] = [getattr(cl, mtd)('Comentário Tratado') for mtd in metodos][-1]

###################
### Vetorização ###
###################
model = Word2Vec(df['Comentário Tratado'].to_numpy(), vector_size=100, window=5, min_count=1, workers=4)
vecs = [(x - np.min(x))/(np.max(x)-np.min(x)) for x in list(df['Comentário Tratado'].apply(lambda x: vectorizeText(model, x)).to_numpy())]
y = df['Número de Reclamações'].values

##########
### NB ###
##########
mNB = MultinomialNB()
parm_grid = {'alpha': np.arange(1e-10, 1.01, 1e-3), 'fit_prior':[True, False]}
mNB_gscv = GridSearchCV(mNB, parm_grid, cv=15)
mNB_gscv.fit(vecs, y)

best_params = mNB_gscv.best_params_
print(best_params)
print(mNB_gscv.best_score_)

modelo = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
modelo.fit(vecs, y)

resultados = cross_val_predict(modelo, vecs, y, verbose=1, cv=15, n_jobs=-1)
features = modelo.n_features_in_

print(f"accuracy: {metrics.accuracy_score(y, resultados)}")
print()
print(metrics.classification_report(y, resultados))
print(f'features: {features}')
print()
print(f'confusion matrix:\n {metrics.confusion_matrix(y, resultados)}')

###########
### KNN ###
###########
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 32, 1), 
              'weights': ['uniform', 'distance'], 
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size':  np.arange(1, 120, 1),
              'p': [1, 2]}
knn_gscv = GridSearchCV(knn, param_grid, cv=15, scoring='accuracy', verbose=1)

knn_gscv.fit(vecs, y)

best_params = knn_gscv.best_params_
print(best_params)
print(knn_gscv.best_score_)

modelo = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], 
                              weights=best_params['weights'], 
                              algorithm=best_params['algorithm'],
                              leaf_size=best_params['leaf_size'],
                              p=best_params['p'])
modelo.fit(vecs, y)

resultados = cross_val_predict(modelo, vecs, y, verbose=1, cv=15, n_jobs=-1)
features = modelo.n_features_in_

print(f"accuracy: {metrics.accuracy_score(y, resultados)}")
print()
print(metrics.classification_report(y, resultados))
print(f'features: {features}')
print()
print(f'confusion matrix:\n {metrics.confusion_matrix(y, resultados)}')

############
### LSTM ###
############
lstm = LSTMClassifier()
param_grid = {'lstm_units': np.arange(100,1000, 50), 'dropout_rate': np.arange(0.0001, 1, 0.001), 'epochs': np.arange(5, 100, 5), 'batch_size': np.arange(16, 129, 16), 'optimizer': ['adam', 'rmsprop']}
lstm_gscv = GridSearchCV(estimator=lstm, param_grid=param_grid, cv=15)
lstm_gscv.fit(vecs, y)
