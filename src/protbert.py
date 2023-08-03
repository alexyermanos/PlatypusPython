from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm 
import torch
class ProtBert():

    """
    Class for the protein Language Model
    """

    def __init__(self, method = "average", file_name = "."):
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

        self.model = BertModel.from_pretrained("Rostlab/prot_bert")

        self.method = method
        self.file = file_name


    def fit_transform(self, sequences:list, batches = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved

        return
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        
        pooler_zero = np.zeros((len(sequences), 1024))
        print("\nUsing the {} method".format(self.method))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                tokenized_sequences = self.tokenizer(sequence[1], return_tensors= 'pt') #return tensors using pytorch
                output = self.model(**tokenized_sequences)
                if self.method == "average":
                    output = torch.mean(output.last_hidden_state, axis = 1)[0]
                
                elif self.method == "pooler":
                    output = output.pooler_output[0]
                elif self.method == "first":
                    output = output.last_hidden_state[:,0][0]
                elif self.method == "last":
                    output = output.last_hidden_state[:,-1][0]
                    
                pooler_zero[sequence[0],:] = output.tolist()
                if sequence[0] % (batch_size+1) == 0:   #Checkpoint save
                    pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv") 

        pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")