Java form built in Netbeans IDE 8.0.2 with jdk-1.8

To run, double click HeartPrediction.jar and fill in all
required information

Author: Brandon Veber
Created: 6-12-2015
Last Modified: 6-21-2015
Tested on Windows 8

Requires:
Python >= 3.4.1:
	Python package dependencies:
		sklearn >= 0.16.1
		numpy   >= 1.9.0
		pandas  >= 0.15.0

Prediction Technical Specifications:
	Data:
		The data is from the UCI Machine Learning Repository Heart Disease Data Set.
		https://archive.ics.uci.edu/ml/datasets/Heart+Disease
		Contains 303 samples with 13 input feauts, and binary target values
 
	Input Features (as seen in GUI form):
		Age - real number
		Sex - categorical
		Chest pain type - categorical
		Resting blood pressure - real number
		Serum cholesterol - real number
		Fasting blood sugar - real number (internally converted to categorical)
		Resting ECG results - categorical
		Maximum achievable heart rate - real number
		Exercised induced angina - categorical
		ST depression induced by exercise relative to rest - real number
		Slope of the peak exercise ST segment - categorical
		Number of major vessels colored by floroscopy - real number
		Thallium heart scan - categorical

	Feature encoding:
		All categorical features are converted to binary using a 
		one-hot encoder

	Feature normalization:
		All real numbers are scaled using a standard scaler (subtract mean
		and divide by standard deviation)

	Machine Learning Algorithm:
		A radial basis kernel SVM classifier is used for making predictions.
		The output is a probability representing the likelihood of the 
		presence of heart disease.

	Expected Accuray:
		The data was tested using a 10-fold cross validation technique.
		The results are:

		Avg Accuracy   Avg Recall     Avg Precision  Avg ROC_AUC
		0.842          0.792          0.863          0.932