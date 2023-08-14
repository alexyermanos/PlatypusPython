#Multi/Binary Classification Keras
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
#For Visualization: 
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from itertools import cycle

#Note: Classification can be Binary of Multi-Class depending on feature labels
#@param embeddings takes a pd.DataFrame of embeddings as input
#@param labels takes a list of feature labels which matches the rows of the embeddings describing the data embedded. 
#Note: embeddings and labels length must equal rows of embeddings. 
#@Arg 'Eval_size' Default is 0.1, input must be between 0-1. 
# The Eval size is the percentage of data used to Evaluate the models performance, 0.1 = 10% of the embeddings. 

# @EXAMPLE:
#import Classification.py as clf
# embeddings = pd.read_csv('embeddings.csv')
# metadata = pd.read_csv('metadata')
# labels = metadata['labels']
# clf.get_Classification(embeddings, labels) 
# or 
# clf.get_Classification(embeddings, labels, Eval_size=0.2)


def get_Classification(embeddings, labels, Eval_size=0.1):
    plt_labels = list(set(labels))
    #make the model and subset Evaluation Data:
    model, history, X_eval, Y_eval = make_Model(embeddings, labels, Eval_size)
    Y_eval = np.array(Y_eval)
    #Evaluate Model Performance:
    Y_pred = get_CM(model, X_eval, Y_eval, plt_labels)
    get_ROC(Y_eval, Y_pred, plt_labels)
    
#Input Embeddings (DF) and labels (Array)
def make_Model(embeddings, labels, Eval_size=0.1):
    X = np.array(embeddings)
    #Encode labels: 
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    if(len(set(labels))>2):
        print('one-hot encoding labels for Multi-class Classification')
        Y = np_utils.to_categorical(Y)
    X = X.astype(float)
    #split test and training:
    X, X_eval, Y, Y_eval = train_test_split(X, Y, test_size = Eval_size)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    #Run model:
    if(len(set(labels))==2):
        print("Running Binary Classification")
        model = Sequential()
        model.add(Dense((len(embeddings.columns)*3), input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size = 32, verbose = 1,
                    epochs = (len(embeddings.columns)*3), validation_data = (X_test, Y_test), shuffle = False)
        _, accuracy = model.evaluate(X_test,Y_test)
        print("Accuracy = ", (accuracy*100),"%")
        return model, history.history, X_eval, Y_eval
    elif(len(set(labels))>2):
        print("Running Multi-class Classification")
        model = Sequential()
        model.add(Dense((len(embeddings.columns)*3), input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(len(set(labels)), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size = 32, verbose = 1,
                    epochs = (len(embeddings.columns)*3), shuffle = False)
        _, accuracy = model.evaluate(X_test,Y_test)
        print("Accuracy = ", (accuracy*100),"%")
        return model, history.history, X_eval, Y_eval
    
def get_CM(model, X_test, Y_act, labels):
    if(len(labels) > 2):
        print('Multi-Class Confusion Matrix')
        Y_pred = model.predict(X_test)
        print('Complete Confusion Matrix')
        cm = confusion_matrix(Y_act.argmax(axis=1),Y_pred.argmax(axis=1))
        cm_df = pd.DataFrame(cm, index = labels, columns = labels)
        plt.figure(figsize=((len(labels)+1),len(labels)))
        sns.heatmap(cm_df, annot=True)
        #Sum confusion Matrices
        mcm = multilabel_confusion_matrix(Y_act.argmax(axis=1),Y_pred.argmax(axis=1))
        cm = [[0,0],[0,0]]
        for i in mcm: 
            cm += i
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= np.unique(Y_act)).plot()
        plt.show()
        return Y_pred
    elif(len(labels)==2):
        print('Binary Confusion Matrix')
        Y_pred = model.predict(X_test)
        cm = confusion_matrix(Y_act.argmax(axis=1), Y_pred.argmax(axis=1), labels = np.unique(Y_act))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= np.unique(Y_act)).plot()
        plt.show()
        return Y_pred

def get_ROC(Y_act, Y_pred, labels):
    #Y_pred = model.predict(X_test)
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
