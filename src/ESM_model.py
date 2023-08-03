from transformers import AutoTokenizer, EsmModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm 
class ESM():

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

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

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
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        print("\nUsing the {} method".format(self.method))
        
        pooler_zero = np.zeros((320, len(sequences)))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                tokenized_sequences = self.tokenizer(sequence[1], return_tensors= 'pt') #return tensors using pytorch
                output = self.model(**tokenized_sequences)

                if self.method == "average":
                    output = torch.mean(output.last_hidden_state, axis = 1)[0]
                
                elif self.method == "pooler":
                    output = output.pooler_output[0]
                
                elif self.method == "last":
                    output = output.last_hidden_state[0,-1,:]

                elif self.method == "first":
                    output = output.last_hidden_state[0,0,:]
                    
                pooler_zero[:,sequence[0]] = output.tolist()
                if sequence[0] % (batch_size) == 0:   #Checkpoint save
                    pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")

        pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")
