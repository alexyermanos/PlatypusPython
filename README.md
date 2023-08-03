# PlatypusPython

## Workflow

The workflow takes as input either a Platypus VGM or a list of sequences in a CSV. Then the sequences are passed into a pretrained language model to make embedings and an optional UMAP for visualization.
#
## Running the workflow

- Create a new enviroment using Python3.8 as:
`conda create -n plm python=3.8`

- Activate the enviroment:
`conda activate plm`

- To install necessary libraries:
`pip install -r requirements.txt`

- Fill in the configuration file with necessary info

- Run the workflow with `snakemake -c (available cores or "all")`

#
### Modes

There are 3 different main modes:
- `VGM` which assumes that a VGM exists in the data folder named data.R which will create the CSV with the sequences.
- `CSV` which assumes that the CSV is ready with the sequences, but the user has to provide the location of the CSV and can provide the name of the column
#

## Argumennts

- `data_path` path with the location of the CSV file or the location of the VGM depending on the mode
- `vgm_colname` column in the vgm that contains the sequences (optional)
- `folder_name` if TRUE then each run output will be stored in a folder with the time and date of execution
- `umap` if True, it also created a UMAP based on the made embedings.

## Arguements per method

### Ablang [\[code\]](https://github.com/oxpig/AbLang/tree/main) 
- `chain`: input chain `heavy`(default) or `light`
- `method`: `seqcoding` returns the average embeding from all tokens

### Antiberty [\[code\]](https://pypi.org/project/antiberty/)
- `method`: strategy with which we get the embedings. Options `last`, `first`, `average`(default)

### Protbert[\[code\]](https://huggingface.co/Rostlab/prot_bert)
- `method`: strategy with which we get the embedings. Options `last`, `first`, `average`(default), `pooler`

### ESM2[\[code\]](https://huggingface.co/docs/transformers/model_doc/esm)
- `method`: strategy with which we get the embedings. Options `last`, `first`, `average`(default)

### Sapiens[\[code\]](https://pypi.org/project/sapiens/)
- `method`: layer from which to extract the embedings per sequence.
- `chain`: input chain `H`(default) or `L`

### TCRBert[\[code\]](https://huggingface.co/wukevin/tcr-bert)
- `method`: layer from which to extract the embedings per sequence.

### How to pass arguements?
In the config.yml, add the line: `arguement` : `value`.\
For example in `Sapiens` the arguements can look like: \
`method`: `average`\
`chain`: `H`\
If you do not provide any argument the default will be passed.
#


## Adding new models

- Create a .py file which will contain the model class in the folder `src/`. Be carefull that the name of the file is not the same as any of the packages that we are using
- Make a model class file like the others (simple example is the ablang_model). 
    - Each class consists of a init function, where you initiallize things, like first making the wanted model and adding it to the self.model variable. 
    - Then it contains a fit_predict which will use the predict function of the model, to get the embedings and print the output in a csv file.
- Then add the import to the make_membedings.py which will use the model.
- The user then can pick which model they want through the congig.yml `model` arguement