import pandas

from sklearn.preprocessing import StandardScaler 

import kfold_template

from sklearn.ensemble import RandomForestClassifier

import numpy

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(['filename'], axis=1)

dataset = dataset.sample(frac=1).reset_index(drop=True)

print(dataset)


target = dataset.iloc[:,-1]
target, key = pandas.factorize(target)
print(key)

data = dataset.iloc[:,:-1]
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype=float))





machine = RandomForestClassifier(criterion="gini", max_depth=100, n_estimators = 300, bootstrap=True, max_features="auto")

results = kfold_template.run_kfold(data, target, 4, machine, 1, 1)

print(results[1])
for i in results[2]:
	print(i)



import keras
from keras import layers
from keras.models import Sequential 

from sklearn.model_selection import KFold
from sklearn import metrics

split_number = 4

kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)

results_accuracy = []
results_confusion = []

for training_index, test_index in kfold_object.split(data):
	data_training = data[training_index]
	target_training = target[training_index]
	data_test = data[test_index]
	target_test = target[test_index]
	
	machine = Sequential()
	machine.add(layers.Dense(256, activation="relu", input_shape=(data_training.shape[1],) ))
	machine.add(layers.Dense(128, activation="relu"))
	machine.add(layers.Dense(64, activation="relu"))
	machine.add(layers.Dense(10, activation="softmax"))
	machine.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

	machine.fit(data_training, target_training, epochs=100)	

	new_target = numpy.argmax(machine.predict(data_test), axis=-1)
	results_accuracy.append(metrics.accuracy_score(target_test, new_target))
	results_confusion.append(metrics.confusion_matrix(target_test, new_target))
	print(results_accuracy)
	for i in results_confusion:
		print(i)



















