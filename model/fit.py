import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel('dinrent (1).xls')
np_dataset = np.array(dataset)
np_y = np_dataset[:,2]
np_x = np_dataset[:,1]
np_x = np_x.reshape(-1, 1)
np_y = np_y.reshape(-1, 1)
model = LinearRegression()
model.fit(np_x, np_y)

print(float(model.predict([[3]])))


with open('Iris_pickle_file_jilie', 'wb') as pkl:
    pickle.dump(model, pkl)