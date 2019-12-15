import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree import BaseDecisionTree
from forest import RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

table = pd.read_csv(r'C:\Users\sadyk\PycharmProjects\University\sdss_redshift.csv')
x = table[list('ugriz')]
y = table['redshift']

x = np.atleast_2d(x)
y = np.atleast_1d(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

forest = RandomForest(x_train, y_train, max_depth=10)
forest.build_forest(x_train, y_train)
y_predict = forest.predict(x_test)
plt.scatter(y_test, y_predict)
plt.plot(plt.xlim(), plt.xlim(), color='m')
plt.savefig('redhift.png')


y_predict_train = forest.predict(x_train)
test = np.sqrt(np.mean(((y_predict - y_test)**2)))
train = np.sqrt(np.mean(((y_predict_train - y_train)**2)))

#---------запись в файл---------------------------
dict = {"train": train, "test": test}
myFile = open("redhsift.json", 'w')
json.dump(dict, myFile)
myFile.close()


# для нахождения оптимума гиперпараметров
params = {'n_estimators': np.arange(1,11), 'max_depth': np.arange(1,11)}
clf = RandomForestRegressor()
clf_grid = GridSearchCV(clf, params, cv=4, n_jobs=-1)
clf_grid.fit(x_train, y_train)
best_params = clf_grid.best_params_
#print(best_params)

#-------------предсказание-------------------------
table = pd.read_csv(r'C:\Users\sadyk\PycharmProjects\University\sdss.csv')
x = table[list('ugriz')]
y_predict = forest.predict(x.values)
x['redshift']=y_predict
x.to_csv(r'C:\Users\sadyk\PycharmProjects\University\sdss_predict.csv')
