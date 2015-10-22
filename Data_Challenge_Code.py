"""
Created on Tue, Oct 6th, 2015

Author: Romano
"""


#%%******************************************************************************
# Importing packages
#******************************************************************************
import numpy as np  #library for matrix-array analysis
import pandas as pd  #library for advanced data analysis
import matplotlib.pyplot as plt #library to plot graphs
import pickle
#******************************************************************************

#%%******************************************************************************
# Ipython Notebook settings
#******************************************************************************
%matplotlib inline
#******************************************************************************

#******************************************************************************
# Define the functions
#******************************************************************************
#--------------------------------------------------

def GET_TrainTest(Feature_DF, Standardize='False', Drop_Unnecessary = 'None', Drop_Cat = 'None', Test_Size = 0.25):

	from sklearn.cross_validation import train_test_split #import package to create the train and test dataset
	from sklearn.preprocessing import StandardScaler as SC #import the standard scaler package to standardize datasets

	if Drop_Unnecessary=='None':
		Drop_Unnecessary = []
		Drop_Unnecessary[0]= Drop_Cat
	#end

	if Standardize == 'True': #if standardization of the predictor features is required
	    Scaler = SC()      
	    X_DS = Scaler.fit_transform(Feature_DF.drop(Drop_Unnecessary, axis=1).values) #extract the features dataset and standardize it
	else: #if no standardization is required
	    X_DS = Feature_DF.drop(Drop_Unnecessary, axis=1).values #extract the features dataset without standardization
	    Scaler = 'None'
	#end
	Y_DS = Feature_DF[Drop_Cat].values #Extracting the CAP damages for train and test dataset
	X_train, X_test, y_train, y_test = train_test_split(X_DS, Y_DS, test_size=Test_Size) #split the dataset into train and test subsets

	return X_train, X_test, y_train, y_test, Scaler, X_DS, Y_DS

#end

#******************************************************************************
def LogReg(X_DS, Y_DS, X_train, X_test, y_train, y_test, Cl_Names = 'None', mask='None',weights='auto'):
#******************************************************************************

	from sklearn.linear_model import LogisticRegression as LogR #import the Logistic Regression module
	from sklearn.metrics import classification_report

	print 'Logistic Regression: Training...' #notify the user about the status of the process 

	LogReg_obj = LogR(C=1e3, class_weight=weights) #create the logistic regression model
	LogReg_obj.fit(X_train, y_train) #fit the logistic model to the train data sets
	Pred_Train = LogReg_obj.predict(X_train) #apply the logistic model to the train dataset
	Pred_Test = LogReg_obj.predict(X_test) #apply the logistic model to the test dataset

	print 'Logistic Regression: Completed!' #notify the user about the status of the process

	labels = len(np.unique(Y_DS)) #extract the labels from the classification classes
	Conf_M = np.zeros((labels,labels), dtype='int') #initialize the confusion matrix for the classification problem
	
	if Cl_Names != 'None':
		target_names = Cl_Names
	else:
		target_names = np.arange(len(np.unique(Y_DS))).astype(str).tolist()
	#end

	Conf_M = CM(y_test, Pred_Test,np.unique(Y_DS)) #calls the confusion matrix routine with the test set and prediction set

	print(classification_report(y_test, Pred_Test, target_names=target_names))  #print the performance indicators on the console

	return LogReg_obj, Conf_M

#end

#******************************************************************************
def RandForest(X_DS, Y_DS, X_train, X_test, y_train, y_test, Cl_Names = 'None', mask='None',Estimators=100):
#******************************************************************************

	from sklearn.ensemble import RandomForestClassifier as RFC #import library for machine learning analysis
	from sklearn.metrics import classification_report

	print 'Random Forest: Training...' #notify the user about the status of the process 

	Random_Forest_obj = RFC(n_estimators=Estimators) #call the Random Forest routing built in
	Random_Forest_obj.fit(X_train, y_train) #fit the logistic model to the train data sets
	Pred_Train = Random_Forest_obj.predict(X_train) #apply the logistic model to the train dataset
	Pred_Test = Random_Forest_obj.predict(X_test) #apply the logistic model to the test dataset

	print 'Random Forest: Completed!' #notify the user about the status of the process

	labels = len(np.unique(Y_DS)) #extract the labels from the classification classes
	Conf_M = np.zeros((labels,labels), dtype='int') #initialize the confusion matrix for the classification problem
	
	if Cl_Names != 'None':
		target_names = Cl_Names
	else:
		target_names = np.arange(len(np.unique(Y_DS))).astype(str).tolist()
	#end

	Conf_M = CM(y_test, Pred_Test,np.unique(Y_DS)) #calls the confusion matrix routine with the test set and prediction set

	print(classification_report(y_test, Pred_Test, target_names=target_names))  #print the performance indicators on the console

	return Random_Forest_obj, Conf_M

#end

#******************************************************************************
def GradBoost(X_DS, Y_DS, X_train, X_test, y_train, y_test, Cl_Names = 'None', mask='None',Max_Depth=3):
#******************************************************************************

	from sklearn.ensemble import GradientBoostingClassifier as GBC #import library for machine learning analysis
	from sklearn.metrics import classification_report

	print 'Gradient Boosting: Training...' #notify the user about the status of the process 

	Gradient_Boosting_obj = GBC(max_depth=Max_Depth) #call the Gradient Boosting routine built in
	Gradient_Boosting_obj.fit(X_train, y_train) #fit the logistic model to the train data sets
	Pred_Train = Gradient_Boosting_obj.predict(X_train) #apply the logistic model to the train dataset
	Pred_Test = Gradient_Boosting_obj.predict(X_test) #apply the logistic model to the test dataset

	print 'Gradient Boosting: Completed!' #notify the user about the status of the process

	labels = len(np.unique(Y_DS)) #extract the labels from the classification classes
	Conf_M = np.zeros((labels,labels), dtype='int') #initialize the confusion matrix for the classification problem
	
	if Cl_Names != 'None':
		target_names = Cl_Names
	else:
		target_names = np.arange(len(np.unique(Y_DS))).astype(str).tolist()
	#end

	Conf_M = CM(y_test, Pred_Test,np.unique(Y_DS)) #calls the confusion matrix routine with the test set and prediction set

	print(classification_report(y_test, Pred_Test, target_names=target_names))  #print the performance indicators on the console

	return Gradient_Boosting_obj, Conf_M

#end

def CrossValidationROC(X,y,Class,Title = ''):

	import numpy as np
	from scipy import interp

	from sklearn import svm, datasets
	from sklearn.metrics import roc_curve, auc
	from sklearn.cross_validation import StratifiedKFold
	from sklearn.ensemble import RandomForestClassifier as RFC #import library for machine learning analysis
	from sklearn.linear_model import LogisticRegression as LogR
	from sklearn.ensemble import GradientBoostingClassifier as GBC

	###############################################################################
	# Define a random state
	random_state = np.random.RandomState(0)

	###############################################################################
	# Classification and ROC analysis

	# Run classifier with cross-validation and plot ROC curves
	cv = StratifiedKFold(y, n_folds=6)
	if Class==1:
		classifier = LogR(C=1e3, class_weight='auto', random_state=random_state)
	#end
	if Class==2:
		classifier = RFC(n_estimators=100, random_state=random_state)
	#end
	if Class==3:
		classifier = GBC(max_depth=5, random_state=random_state)
	#end

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	for i, (train, test) in enumerate(cv):
	    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
	    # Compute ROC curve and area the curve
	    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	    mean_tpr += interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    roc_auc = auc(fpr, tpr)
	    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
	#end

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if Class==1:
		plt.title('ROC Logistic Regression')
		plt.legend(loc="lower right")
	#end
	if Class==2:
		plt.title('ROC Random Forest')
		plt.legend(loc="lower right")
	#end
	if Class==3:
		plt.title('ROC Gradient Boosting')
		plt.legend(loc="lower right")
	#end
#end


def HeatmapCorr(Dataframe):

	plt.figure()

	import seaborn as sns

	# Load the datset of correlations between cortical brain networks
	corrmat = Dataframe.corr()

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 9))

	# Draw the heatmap using seaborn
	sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True)

#end


#******************************************************************************
# Main Program
#******************************************************************************


#-------------------------------------------
# Define the paths
#-------------------------------------------
Input_Path = '/media/romano/Data1/Data Science/Insight 2015/Data Challenge/'
Output_Path = Input_Path
#******************************************************************************

#-------------------------------------------
# Create the dataframes
#-------------------------------------------

Breast_Cancer_DF = pd.read_csv(Input_Path + 'breast-cancer-wisconsin.csv')

#*********************************************

#-------------------------------------------
# Manipulating the Outcome
#-------------------------------------------
Breast_Cancer_DF['Class'][Breast_Cancer_DF['Class']==2] = 0 #assign 0 to benign
Breast_Cancer_DF['Class'][Breast_Cancer_DF['Class']==4] = 1 #assign 1 to malignant

#******************************************************************************
# Plotting Dataset characteristics
#******************************************************************************

#-------------------------------------------
# Plotting histograms
#-------------------------------------------

Labelsize = 16
Titlesize = 16
Ticksize = 12
Transparency = 0.7
Fig_counter = 1

for ik, Key_name in enumerate(Breast_Cancer_DF.keys().tolist()[1:10]):
    plt.figure(Fig_counter)
    Benign = Breast_Cancer_DF[Key_name][Breast_Cancer_DF.Class==0]
    Malignant = Breast_Cancer_DF[Key_name][Breast_Cancer_DF.Class==1]
    Title = 'Cancer incidence vs ' + Key_name
    fig = Benign.plot(kind='kde',color='green',linewidth=2) 
    Malignant.plot(kind='kde',color='red',ax=fig,linewidth=2)
    Bins = map(float,range(0, 11))
    Bins = [x+0.1 for x in Bins]
    _ = fig.hist(Benign.values, normed=True, color='green', alpha=Transparency, bins=Bins) #N_bins[eb])
    _ = fig.hist(Malignant.values, normed=True, color='red', alpha=Transparency, bins=Bins) #N_bins[eb])
    fig.legend(['Benign','Malignant'], fontsize=Labelsize)
    fig.set_xlabel('Categorical Value', fontsize=Labelsize)
    fig.set_ylabel(Key_name, fontsize=Labelsize)
    fig.set_title(Title, fontsize=Titlesize)
    fig.tick_params(axis='x', labelsize=Ticksize)
    fig.tick_params(axis='y', labelsize=Ticksize)
    fig.set_xlim([0,10])
    Fig_counter = Fig_counter + 1
#end

#--------------------------------------------------

#-------------------------------------------
# Plotting the scatter cross-correlation graphs
#-------------------------------------------
from pandas.tools.plotting import scatter_matrix
Scatter_M_DF = Breast_Cancer_DF.drop(['ID','Class'], axis=1)
scatter_matrix(Scatter_M_DF, alpha=0.2, diagonal='kde')

#-------------------------------------------

#-------------------------------------------
# Plotting the cross-correlation matrix as heatmap
#-------------------------------------------

Scatter_M_DF_C = Breast_Cancer_DF[Breast_Cancer_DF.Class==1].drop(['ID','Class'],axis=1)
HeatmapCorr(Scatter_M_DF_C)
HeatmapCorr(Scatter_M_DF)
#-------------------------------------------

#-------------------------------------------
# Working with missing data
#-------------------------------------------
Breast_Cancer_Filled_DF = Breast_Cancer_DF.copy()
Breast_Cancer_Filled_DF.Bare_Nuclei[Breast_Cancer_DF.Bare_Nuclei.isnull()==True]=Breast_Cancer_DF.Bare_Nuclei.mean()
#-------------------------------------------

#-------------------------------------------
# Train-test split
#-------------------------------------------
Split_Size = 0.25
X_train, X_test, y_train, y_test, Scaler, X_DS, Y_DS = GET_TrainTest(Breast_Cancer_Filled_DF, Standardize='False', Drop_Unnecessary = ['ID', 'Class'], Drop_Cat = 'Class', Test_Size = Split_Size) #split the dataset into train and test subsets

#-------------------------------------------
# Plotting cross-validated ROC Curves
#-------------------------------------------
plt.figure()
CrossValidationROC(X_DS,Y_DS,1) #Using Logistic Regression
plt.figure()
CrossValidationROC(X_DS,Y_DS,2) #Using Random Forest
plt.figure()
CrossValidationROC(X_DS,Y_DS,3) #Using Gradient Boosting
#-------------------------------------------

#********************************************************************************

