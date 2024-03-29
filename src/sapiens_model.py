import sapiens
import pandas as pd
import numpy as np
import pickle as pkl
class Sapiens():

    """
    Class for the protein Model Sapiens
    Author: Aurora
    """

    def __init__(self, chain_type="H", method="average", file_name = "."):
        """
        Creates the instance of the language model instance

        parameters
        ----------

        chain_type: `str`
        `L` or `H` whether the input is from light or heavy chains resprectively
        
        method: `str`
        Layer that we want the embedings from

        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.chain = chain_type
        if isinstance (method,int):
            self.layer = method
        elif method == "average":
            self.layer = None
        self.file = file_name

    def fit_transform(self, sequences):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        Column with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """

        if self.layer == None:
            print("Using the average layer")
            output = sequences.apply(lambda seq: pd.Series(np.mean(sapiens.predict_sequence_embedding(seq, chain_type=self.chain),axis = 0)))
            output.to_csv("outfiles/"+self.file+"/embeddings.csv") #We have one embeded sequence per row
        else:
            print("\nUsing the {} layer".format(self.layer))
            output = sequences.apply(lambda seq: pd.Series(sapiens.predict_sequence_embedding(seq, chain_type=self.chain, layer=self.layer)))
            output.to_csv("outfiles/"+self.file+"/embeddings.csv") #We have one embeded sequence per row