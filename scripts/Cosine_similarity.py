#Init
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

#Takes two inputs 
#@param embeddings takes a pd.DataFrame of embeddings as input.
#@param labels takes a list of feature labels which matches the rows of the embeddings describing the data embedded. 
#Note: embeddings and labels length must equal rows of embeddings. 

# @EXAMPLE:
# import Cosine_similarity.py as CosS
# embeddings = pd.read_csv('embeddings.csv')
# metadata = pd.read_csv('metadata')
# labels = metadata['labels']
# CosS.cos_sim(embeddings, labels) 


#Function:
def cos_sim(embeddings, labels):
    embeddings['temp'] = labels
    embeddings_subs = []
    uniq_labs = np.unique(labels)
    #Subset data
    for lab in uniq_labs:
        temp = embeddings[embeddings['temp'] == lab]
        temp = temp.iloc[:, :-2]
        embeddings_subs.append(temp.to_numpy())
    results = []
    #Calc cos similarites for every group vs ever group
    for i in embeddings_subs:
        for c in embeddings_subs: 
            temp1 = np.mean(i, axis=0).reshape(1, -1)
            temp2 = np.mean(c, axis=0).reshape(1, -1)
            results.append(cosine_similarity(temp1, temp2))
    results = pd.DataFrame(data=[results[i:i+len(uniq_labs)] for i in range(0, len(results), len(uniq_labs))], columns=uniq_labs, index = uniq_labs)
    results = results.applymap(lambda x: x[0])
    results = results.applymap(lambda x: x[0])
    #show table of cosine similarities
    print(results)
    # Create a heatmap of the cosine similarities
    plt.figure(figsize=(len(uniq_labs)*2, len(uniq_labs)*1.5))
    plt.imshow(results, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Log Cos Scale')
    plt.title('Feature Cosine Similarity')
    plt.xticks(range(len(results.columns)), results.columns, rotation=45)
    plt.yticks(range(len(results.index)), results.index)
    for i in range(len(results.index)):
        for c in range(len(results.columns)):
            plt.text(c, i, round(results.iloc[i, c], 4) , ha='center', va='center', color='black', fontsize=12)
    plt.show()
