import numpy as np	
	
	
def confusion_matrix(labels, predictions):

	cms = {}

	for label in range(3):

		# Inicializar la matriz de confusión 2x2 para la clase actual
		cm = np.zeros((2, 2), dtype=int)
		
		# Iterar sobre las listas de valores reales y predichos
		for true, pred in zip(labels, predictions):
			if true == label and pred == label:
				cm[0, 0] += 1  # TP
			elif true == label and pred != label:
				cm[1, 0] += 1  # FN
			elif true != label and pred == label:
				cm[0, 1] += 1  # FP
			elif true != label and pred != label:
				cm[1, 1] += 1  # TN
		
		# Almacenar la matriz de confusión 2x2 en el diccionario
		cms[label] = cm

	return cms


def calculate_accuracy(cms):

	accuracy_list = []

	for _, cm in cms.items():
		numerator = cm[0, 0] # TPi
		denominator = cm[0, 0] + cm[1, 0] + cm[0, 1] + cm[1, 1] # TPi + FNi + FPi + TNi

		if denominator == 0:
			accuracy_list.append(0)
		else:
			accuracy_list.append((numerator / denominator) * 100)

	return accuracy_list



def calculate_precision_recall(cms):

	precision_list = []
	recall_list = []

	for _, cm in cms.items():
		numerator_prec = cm[0, 0] # TPi
		denominator_prec = cm[0, 0] + cm[0, 1] # TPi + FPi

		if denominator_prec == 0:
			precision_list.append(0)
		else:
			precision_list.append((numerator_prec / denominator_prec))


		numerator_rec = cm[0, 0] # TPi
		denominator_rec = cm[0, 0] + cm[1, 0] # TPi + FNi

		if denominator_rec == 0:
			recall_list.append(0)
		else:
			recall_list.append((numerator_rec / denominator_rec))

	return precision_list, recall_list




def calculate_f1(precision_list, recall_list):

	F1_list = []

	for i in range(len(precision_list)):
		numerator = precision_list[i] * recall_list[i]
		denominator = precision_list[i] + recall_list[i]

		if denominator == 0:
			F1_list.append(0)
		else:
			F1_list.append( 2 * ( numerator / denominator ) )

	return F1_list

