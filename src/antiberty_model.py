from antiberty import AntiBERTyRunner
import torch
import pandas as pd
class Antiberty():

    """
    Class for the protein Model Antiberty
    """

    def __init__(self, token = "average", file_name = "."):
        """
        Creates the instance of the language model instance; either light or heavy
        
        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        self.model = AntiBERTyRunner()
        self.token = token
        self.file = file_name

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
        out_list = []
        print("Using the {} token\n".format(self.token))
        output = self.model.embed(sequences)
        if self.token == "average":
            for embedding in output: #Taking the average for each of the aminoacid values
                out_list.append(torch.mean(embedding, axis = 0).tolist())
        elif self.token == "last":
            for embedding in output: #Take the last token
                out_list.append(embedding[-1].tolist())
        elif self.token == "first":
            for embedding in output: #Take only CLS
                out_list.append(embedding[0].tolist())
        pd.DataFrame(out_list).to_csv("outfiles/"+self.file+"/embeddings.csv")
