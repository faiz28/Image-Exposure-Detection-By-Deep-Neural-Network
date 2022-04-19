from tensorflow.keras.metrics import MeanIoU, Recall, Precision, Accuracy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam,schedules
from sklearn.metrics import confusion_matrix
import numpy as np
import load_image


DIR = './Small-Size-Convolution-Maxpooling-model/'
modelpath = DIR + 'Small_CNN_Classifier.hdf5'

print("data")

lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

sgd = SGD(learning_rate=lr_schedule)

# def main():
# #     print("data23")

# # testInSet, testOutSet = prepare_testSet()
# x_train, y_train, x_test, y_test = load_image.load_image_data()
# print("%s  %s"%(x_test.shape[0], y_test.shape[0])) 
# '''	Load pre-trained model.	'''
# # modelFileName = modelDir + 'modelName.hdf5'
# model = load_model(modelpath, compile = False)

# '''	Set Metrics.	'''	
# model.compile(loss = 'mse', optimizer = sgd, metrics = [ Accuracy(), Recall(), Precision()])

# '''	Test Model. '''
# loss, acc, precision, recall = model.evaluate(x_test, y_test, batch_size = 128)

# print("\n%s %s %s %s\n"%(loss,acc,precision,recall))

# # if __name__ == '_main__':
# # 	main()

def main():
	train_x,train_Y, testInSet, testOutSet =load_image.load_image_data()
 
	# indices = np.argwhere(np.argwhere(testOutSet) == 0)
	# testInNormal = testInSet[indices]
	# testOutNormal = testOutSet[indices]
	# indices = np.argwhere(testOutSet == 1)
	# testInOver = testInSet[indices]
	# testOutOver = testOutSet[indices]
	# indices = np.argwhere(testOutSet == 2)
	# testInUnder = testInSet[indices]
	# testOutUnder = testOutSet[indices]

	'''	Load pre-trained model.	'''
	# modelFileName = modelDir + 'modelName.hdf5'
	model = load_model(modelpath, compile = False)

	'''	Set Metrics.	'''	
	model.compile(loss = 'mse', optimizer = sgd, metrics = ['accuracy', Recall(), Precision()])

	'''	Test Model. '''
	lossT, accT, precisionT, recallT = model.evaluate(testInSet, testOutSet)
	# lossN, accN, precisionN, recallN = model.evaluate(testInNormal, testOutNormal)
	# lossO, accO, precisionO, recallO = model.evaluate(testInOver, testOutOver)
	# lossU, accU, precisionU, recallU = model.evaluate(testInUnder, testOutUnder)
 
	print("\n\nlossT = %s, accT = %s,  precisionT = %s, recallT = %s "%(lossT, accT, precisionT, recallT))
	# print("lossN = %s, accN = %s,  precisionN = %s, recallN = %s "%(lossN, accN, precisionN, recallN))
	# print("lossO = %s, accO = %s,  precisionO = %s, recallO = %s "%(lossO, accO, precisionO, recallO))
	# print("lossU = %s, accU = %s,  precisionU = %s, recallU = %s "%(lossU, accU, precisionU, recallU))
		
	predicted_output = []
	prediction = model.predict(testInSet)
	for i in prediction:
		predicted_output.append(np.argmax(i)) 

	actual_output = []
	for i in testOutSet:
		actual_output.append(np.argmax(i)) 
	con_mat = confusion_matrix(actual_output, predicted_output)
	for i in con_mat:
		print(i)



# if __name__ == '__main__':
print("Hello")
main()

# predicted_output = []
# prediction = model.predict(testInSet)
# for i in prediction:
#     predicted_output.append(np.argmax(i)) 

# actual_output = []
# for i in y_test:
#     actual_output.append(np.argmax(i)) 
# con_mat = confusion_matrix(actual_output, predicted_output)
# for i in con_mat:
#     print(i)

