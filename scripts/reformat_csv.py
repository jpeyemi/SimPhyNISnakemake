# scripts/reformat_csv.py
import sys
import pandas as pd
import re

def reformat_string_for_filepath(s):
    replacements = {
        ' ': '_', '\\': '', '/': '', ':': '', '*': '',
        '?': '', '"': '', '<': '', '>': '', '|': '', '.':'_', '~':'',
    }
    for key, value in replacements.items():
        s = s.replace(key, value)
    return re.sub(r'[^a-zA-Z0-9_.-]', '', s)

def reformat_columns(input_csv, output_csv):
    data = pd.read_csv(input_csv, index_col=0)
    data.columns = [reformat_string_for_filepath(col) for col in data.columns]
    data[data > 0] = 1
    data.fillna(0, inplace=True)
    data = data.astype(int)
    data.to_csv(output_csv)

if __name__ == "__main__":
    reformat_columns(sys.argv[1], sys.argv[2])
