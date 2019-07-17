#Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

proq = pd.read_csv('ProQDock.csv')

columns_of_features = ['rGb','nBSA', 'Fintres', 'Sc', 'EC','ProQ', 'zrank', 'zrank2', 'Isc','rTs', 'Erep', 'Etmr', 'CPM', 'Ld', 'CPscore']
columns_of_target = ['DockQ']

proq_data = proq[columns_of_features]
proq_target = proq[columns_of_target]

data_train, data_test, target_train, target_test = train_test_split(proq_data, proq_target, test_size=0.2, random_state = 0)

#print (target_train.shape)
#print (target_test.shape)
log_reg = LogisticRegression(solver ='lbfgs', multi_class = 'multinomial').fit(data_train, target_train)
predictions = log_reg.predict(data_test)
print (predictions)

acc = log_reg.score(data_test, target_test)
print (acc)


