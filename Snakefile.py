##########################
# SNAKEFILE FOR SimPhyNI #
##########################

""" PRE-SNAKEMAKE """

sampleFile = 'samples_example.csv' #run_type key: 0 - All against All, 1 - First against All
intermediariesDirectory = '' #EX: '/home/iobal/orcd/c7/scratch/iobal/AssociationSnakemake/' 
# Empty string will put intermediaries in the same file systems as outputs
# Must end with /
import sys
SCRIPTS_DIRECTORY = "./scripts"
sys.path.insert(0, SCRIPTS_DIRECTORY)


import os
import pandas as pd
import shutil

samples = pd.read_csv(sampleFile)
SAMPLE_ls = samples['Sample']
OBS_ls = samples['Obs Path']
TREE_ls = samples['Tree Path']
RUNTYPE = samples['Run Type']

current_directory = os.getcwd()


def copy_files_to_inputs(file_paths, name):
    input_dir = "inputs/"
    os.makedirs(input_dir, exist_ok=True)
    for file_path, n in zip(file_paths,name):
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(input_dir, n + '.' + file_name.split('.')[-1])
        if not os.path.exists(destination_path):
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        else:
            print(f"File {file_name} already exists in {input_dir}, skipping copy.")

copy_files_to_inputs(OBS_ls, SAMPLE_ls)
copy_files_to_inputs(TREE_ls, SAMPLE_ls)

run_dict = dict(zip(SAMPLE_ls,RUNTYPE))


''' SNAKEMAKE '''

rule all:
    input:
        expand("0-formatting/{sample}/{sample}.csv", zip, sample = SAMPLE_ls),
        expand("0-formatting/{sample}/{sample}.nwk", zip, sample = SAMPLE_ls),
        expand("1-PastML/{sample}/out.txt", sample = SAMPLE_ls),
        expand("2-Events/{sample}/pastmlout.csv", sample = SAMPLE_ls),
        expand("3-Objects/{sample}/sim_kde.pkl", sample = SAMPLE_ls),


rule reformat_csv:
    input:
        inp = 'inputs/{sample}.csv'
    output:
        out = '0-formatting/{sample}/{sample}.csv'
    conda:
        'envs/simphyni.yaml'
    shell:
        'python scripts/reformat_csv.py {input.inp} {output.out}'

rule reformat_tree:
    input:
        inp = 'inputs/{sample}.nwk'
    output:
        out = '0-formatting/{sample}/{sample}.nwk'
    conda:
        'envs/simphyni.yaml'
    shell:
        'python scripts/reformat_tree.py {input.inp} {output.out}'


rule pastml:
    input:
        inputsFile = rules.reformat_csv.output.out,
        tree = rules.reformat_tree.output.out
    output:
        outfile = "1-PastML/{sample}/out.txt",
    params:
        outdir = lambda wildcards: f"{intermediariesDirectory}1-PastML/{wildcards.sample}",
        max_workers = 64
    conda:
        "envs/simphyni.yaml"
    shell:
        "python scripts/pastml.py \
            --inputs_file {input.inputsFile} \
            --tree_file {input.tree} \
            --outdir {params.outdir} \
            --max_workers {params.max_workers} \
            --summary_file {output.outfile} \
        "

rule aggregatepastml:
    input:
        inputsFile = rules.pastml.input.inputsFile,
        tree = rules.pastml.input.tree,
        file = rules.pastml.output.outfile,
    output:
        annotation = "2-Events/{sample}/pastmlout.csv"
    params:
        pastml_folder = rules.pastml.params.outdir,
    conda:
        'envs/simphyni.yaml'
    shell:
        "python scripts/GL_tab.py {input.inputsFile} {input.tree} {params.pastml_folder} {output.annotation}"

rule SimPhyNI:
    input:
        pastml = "2-Events/{sample}/pastmlout.csv",
        systems = "0-formatting/{sample}/{sample}.csv",
        tree = "0-formatting/{sample}/{sample}.nwk"
    output:
        annotation = "3-Objects/{sample}/sim_kde.pkl"
    params:
        outdir = "3-Objects/{sample}/",
        runtype = lambda wildcards: run_dict.get(wildcards.sample, 0)
    conda:
        'envs/simphyni.yaml'
    shell:
        "python scripts/runSimPhyNI.py \
            -p {input.pastml} \
            -s {input.systems} \
            -t {input.tree} \
            -o {params.outdir} \
            -r {params.runtype} \
        "


