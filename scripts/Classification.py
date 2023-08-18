#Multi/Binary Classification
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
#For Visualization: 
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from itertools import cycle
#For Balancing and Scaling: 
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

#This Classification function is used to Classify output embeddings from BERT models. 
#Note: Classification can be Binary of Multi-Class depending on feature labels
#@param embeddings takes a pd.DataFrame of embeddings as input.
#@param labels takes a list of feature labels which matches the rows of the embeddings describing the data embedded. 
#Note: embeddings and labels length must equal rows of embeddings. 
#@param 'eval_size' Default is 0.1, input must be between 0-1. 
#The eval size is the percentage of data used to Evaluate the models performance, 0.1 = 10% of the embeddings. 
#@param 'balancing' takes a boolean as input: default is True.
#Balacing does RandomOverSampling of all minority classes so that the number of minority equals majority class.
#@param 'epochs' takes a numeric input: Default is 3 times the number of embedding columns
#Epochs dictates how many passes the model will undergo during configuration, used to update weight, bias, and parameter optimization.
#@param 'nodes' takes a numeric input: Default is 1/2 the number of embedding columns.
#Influences how many nodes will consist in the single layered neural network. 
#@param 'batch_size' takes a numeric value: Default is 32.
#the batch size determines how many training samples are processed together. 
#@param 'scaling' takes boolean input:
#Uses the sklearn StandardScaler function to scale data

# @EXAMPLE:
#import Classification.py as clf
# embeddings = pd.read_csv('embeddings.csv')
# metadata = pd.read_csv('metadata')
# labels = metadata['labels']
# clf.get_Classification(embeddings, labels) 
# or 
# clf.get_Classification(embeddings, labels, balancing=False, eval_size=0.2, epochs = 200, nodes = 100, batch_size= 16, scaling=False)


def get_Classification(embeddings, labels, balancing=False, eval_size=0.1, epochs=0, nodes=0, batch_size=32, scaling=True):
    #Default epochs is three times the number of columns 
    if(epochs == 0): 
        epochs = len(embeddings.columns)*3
    #Default nodes is one half the number of columns 
    if(nodes == 0):
        nodes = int(len(embeddings.columns)/2)
    plt_labels = list(np.unique(labels))
    #Make Model and Subset Evaluation Data
    model, X_eval, Y_eval = make_NN(embeddings, labels, balancing = balancing, eval_size = eval_size, epochs = epochs,
                                                nodes = nodes, batch_size = batch_size, scale = scaling)
    Y_eval = np.array(Y_eval)
    #Run evalauation subset on model and make confusion matrices
    Y_pred = model.predict(X_eval)
    get_CM(Y_eval.argmax(axis=1), Y_pred.argmax(axis=1), plt_labels)
    #Get ROC_AUC curves (Macro, Micro, and each class vs rest) 
    get_ROC(Y_eval, Y_pred, plt_labels)
    #Make SVM and get Confusion matrices
    model_SVM, X_eval, Y_eval = make_SVM(embeddings, labels, balancing, scale=scaling)
    Y_pred = model_SVM.predict(X_eval)
    get_CM(Y_eval, Y_pred, plt_labels)
    
#Input Embeddings (DF) and labels (Array)
def make_NN(embeddings, labels, balancing, eval_size, epochs, nodes, batch_size, scale):
    X = np.array(embeddings)
    #Encode labels: 
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    if(len(set(labels))>2):
        #print('one-hot encoding labels for Multi-class Classification')
        Y = np_utils.to_categorical(Y)
    X = X.astype(float)
    #Split evaluation sets:
    X, X_eval, Y, Y_eval = train_test_split(X, Y, test_size = eval_size, random_state = 42)
    #Optional Balancing: Random Over Sampling of all classes but highest
    if(balancing == True):
            oversample = RandomOverSampler(sampling_strategy='not majority')
            X, Y = oversample.fit_resample(X, Y)
    #Split training and test sets: 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #Optional Scale: Sklearn StandardScaler
    if(scale == True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_eval = scaler.transform(X_eval)
    #Run Model:
    #If Binary Classification:
    if(len(set(labels))==2):
        print("Running Binary Classification")
        model = Sequential()
        model.add(Dense(nodes, input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size = batch_size, verbose = 1,
                    epochs = epochs, validation_data = (X_test, Y_test), shuffle = False)
        _, accuracy = model.evaluate(X_test,Y_test)
        print("Accuracy = ", (accuracy*100),"%")
        return model, X_eval, Y_eval
    #If Multi-class Classification: 
    elif(len(set(labels))>2):
        print("Running Multi-class Classification")
        model = Sequential()
        model.add(Dense(nodes, input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(len(set(labels)), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size = batch_size, verbose = 1,
                    epochs = epochs, shuffle = False)
        _, accuracy = model.evaluate(X_test,Y_test)
        print("Accuracy = ", (accuracy*100),"%")
        return model, X_eval, Y_eval
    
def get_CM(Y_act, Y_pred, labels):
    if(len(labels) > 2):
        print('Multi-Class Confusion Matrix')
        #print('Complete Confusion Matrix')
        cm = confusion_matrix(Y_act, Y_pred)
        #print('cm' ,cm)
        cm_df = pd.DataFrame(cm, index = labels, columns = labels)
        #print('cm_df ',cm_df)
        #Sum confusion Matrices
        mcm = multilabel_confusion_matrix(Y_act,Y_pred)
        cm = [[0,0],[0,0]]
        for i in mcm: 
            cm += i
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= np.unique(Y_act)).plot()
        plt.show()
    elif(len(labels)==2):
        print('Binary Confusion Matrix')
        cm = confusion_matrix(Y_act, Y_pred, labels = np.unique(Y_act))
        #print('cm ',cm)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= np.unique(Y_act)).plot()
        plt.show()

def get_ROC(Y_act, Y_pred, labels):
    if(len(labels)==2):
        #Binary ROC
        print('Binary')
        fpr, tpr, thresholds = roc_curve(Y_act, Y_pred)
        plt.figure(1)
        plt.plot([0,1],[0,1], 'y--')
        plt.plot(fpr,tpr,marker='.')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.show()
    elif(len(labels) > 2):
        #Multi-class ROC
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_act.ravel(), Y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #Compute Class ROC/AUCs 
        for i in range(len(Y_act[1])):
            fpr[i], tpr[i], _ = roc_curve(Y_act[:, i], Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            #print("roc_auc for", labels[i]," is ",roc_auc[i])
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(len(Y_act[1])):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
        # Macro-Average it and compute AUC
        mean_tpr /= len(list(set(labels)))
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        #Plot Micro-average + Macro-average + Class performances, Respectively:
        #print('roc_auc Micro', roc_auc['micro'])
        #print('roc_auc Macro', roc_auc['macro'])
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        colors_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        colors_list = colors_list[:len(labels)+2]
        colors = cycle(colors_list)
        for class_id, color in zip(range(len(labels)), colors):
            RocCurveDisplay.from_predictions(
                Y_act[:, class_id],
                Y_pred[:, class_id],
                name=f"ROC curve for {labels[class_id]}",
                color=color,
                ax=ax,
            )
        plt.axis("square")
        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")
        plt.title("Receiver Operating Characteristic\nto One vs All Multi=class")
        plt.legend()
        plt.show()

def make_SVM(embeddings, labels, balancing, scale): 
    X = np.array(embeddings)
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    X = X.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #Optional Balancing:
    if(balancing == True):
        oversample = RandomOverSampler(sampling_strategy='not majority')
        X, Y = oversample.fit_resample(X, Y)
    #Optional Scaling:
    if(scale == True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if(len(labels) > 2):
        print('Multi-class SVM')
        model = SVC(decision_function_shape='ovo')
        model.fit(X_train, Y_train)
        print(f'SVM Training Accuracy - :{model.score(X_train, Y_train):.3f}')
        print(f'SVM Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return model, X_test, Y_test
    elif(len(labels) == 2):
        print('Binary SVM')
        model = SVC(gamma = 'auto')
        model.fit(X_train, Y_train)
        print(f'SVM Training Accuracy - :{model.score(X_train, Y_train):.3f}')
        print(f'SVM Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return model, X_test, Y_test
