import ablang
import pandas as pd
import pickle as pkl
class Ablang():

    """
    Class for the protein Model Ablang
    """

    def __init__(self, chain = "heavy",file_name = ".", method = "seqcoding"):
        """
        Creates the instance of the language model instance; either light or heavy
        
        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.model = ablang.pretrained(chain)
        #dont update the weights
        self.model.freeze()

        self.file = file_name
        self.mode = method
    


    def fit_transform(self, sequences):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """
        print(sequences)
        output = self.model(sequences, mode=self.mode)
        if self.mode == "seqcoding":
            #The embeddings are made my averaging across all residues    
            pd.DataFrame(output).to_csv("outfiles/"+self.file+"/embeddings.csv")