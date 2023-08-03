import sys
import src.utils as utils
import yaml
import os
import time
import pprint
#Read the file with parameters
with open("config.yml", "r") as f:
    settings = yaml.safe_load(f)

if not os.path.exists("outfiles"):
    os.mkdir("outfiles")

if "folder_name" in settings:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir("outfiles/"+timestr)
    pprint.pprint(settings, stream=open("outfiles/"+timestr+"/"+"run_info.txt","w"))
else:
    timestr="."

# Choose the sequence name you want to read the csv file
if("csv_colname" in settings):
    csv_colname = settings["csv_colname"]
else:
    csv_colname = "sequence"

mode = [settings["mode"]]

model = settings["model"]

if "CSV" in mode:
    if "data_path" in settings:
        input_file_plm = settings["data_path"]
    else:
        print("Please provide the CSV location in the data_path variable")
else:
    input_file_plm = "outfiles/{}/clones.csv".format(timestr)

if "umap" in settings and settings["umap"] == True:
    output = ["outfiles/{}/embeddings.csv".format(timestr),"outfiles/{}/UMAP.pdf".format(timestr)]
else:
    output = "outfiles/{}/embeddings.csv".format(timestr)

rule all:
    input:
        output

if("CSV" not in mode):
    rule prepare_for_embeddings_no_AB:
        input:
            settings["data_path"]
        output:
            input_file_plm
        shell:
            "Rscript scripts/no_ab_prepare_embeddings.R "+ timestr

rule make_embeddings:
    input:
        input_file_plm
    output:
        "outfiles/{}/embeddings.csv".format(timestr)
    shell:
        "python scripts/make_embeddings.py {input_file_plm} {csv_colname} {timestr}"

rule visualize:
    input:
         "outfiles/{}/embeddings.csv".format(timestr)
    output:
        "outfiles/{}/UMAP.pdf".format(timestr)
    shell:
        "python scripts/make_umap.py {timestr}"
