import os.path
import subprocess
import torch
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from src import __path__ as src_path
import warnings
varna_path = f"{src_path[0]}/tools/varna/VARNAv3-93.jar"
draw_call = f"export DATAPATH={src_path[0]}/tools/RNAstructure/data_tables;  {src_path[0]}/tools/RNAstructure/draw -c -u --svg -n 1"

NT_DICT = {
    "R": ["G", "A"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
    "N": ["G", "A", "C", "U"],
}

VOCABULARY = ["A", "C", "G", "U"]

def write_ct(filename, seq, seq_id, bp):
    # Write the base pairs to a .ct file
    '''
    seq: the sequence
    pairs: the base pairs
    filename: the filename
    '''
    bp_dict = {}
    for i in bp:
        bp_dict[bp[0]] = i[1]
        bp_dict[bp[1]] = i[0]
    with open(filename, "w") as f:
        f.write(f"{len(seq)}\n {seq_id}\n")
        for k, n in enumerate(seq):
            f.write(f"{k + 1} {n} {k} {k + 2} {bp_dict.get(k + 1, 0)} {k + 1}\n")

def valid_sequence(seq):
    """Check if sequence is valid"""
    return set(seq.upper()) <= (set(NT_DICT.keys()).union(set(VOCABULARY)))

def validate_file(filename):
    # Validate the file
    '''
    filename: the filename
    '''
    if os.path.splitext(filename)[1] == ".fasta":
        table = []
        with open(filename) as f:
            row = []
            for line in f:
                if line.startswith(">"):
                    if row:
                        table.append(row)
                        row = []
                    row.append(line[1:].strip())
                else:
                    if len(row) == 1:
                        row.append(line.strip())
                        if not valid_sequence(row[-1]):
                            raise ValueError(f"Invalid characters")
                    else:  # struct
                        row.append(line.strip()[
                                   :len(row[1])])
        if row:
            table.append(row)
        filename = filename.replace(".fasta", ".csv")
        if len(table[-1]) == 2:
            columns = ["id", "sequence"]
        else:
            columns = ["id", "sequence", "dotbracket"]
        pd.DataFrame(table, columns=columns).to_csv(filename, index=False)

    return filename


def ct_to_dot(filename):
    # Convert the .ct file to dotbracket notation
    '''
    filename: the filename
    return: the dotbracket notation
    '''
    if not os.path.isfile(filename):
        raise ValueError(f".ct file does not exist")
    dot_bracket = ""
    CT2DOT_CALL = f"export DATAPATH={src_path[0]}/tools/RNAstructure/data_tables; {src_path[0]}/tools/RNAstructure/ct2dot"

    if CT2DOT_CALL:
        subprocess.run(f"{CT2DOT_CALL} {filename}", shell=True, capture_output=True)
        try:
            dot_bracket = open("tmp.dot").readlines()[2].strip()
            os.remove("tmp.dot")
        except FileNotFoundError:
            print("Error in dotbracket conversion")
    else:
        print("Dotbracket conversion only available in Linux")
    return dot_bracket

def dot2png(png_file, sequence, dotbracket, resolution=20):
    # Convert the dotbracket notation to a png file
    '''
    png_file: the png file
    sequence: the sequence
    dotbracket: the dotbracket notation
    '''
    try:
        subprocess.run("java -version", shell=True, check=True, capture_output=True)
        subprocess.run(
            f'java -cp {varna_path} fr.orsay.lri.varna.applications.VARNAcmd -sequenceDBN {sequence} -structureDBN "{dotbracket}" -o  {png_file} -resolution {resolution}',
            shell=True)
    except:
        warnings.warn("Java Runtime Environment failed trying to run VARNA. Check if it is installed.")


def ct2svg(ct_file, svg_file):
    # Convert the .ct file to a svg file
    '''
    ct_file: the .ct file
    svg_file: the svg file
    '''
    subprocess.run(
        f"{draw_call} {ct_file} {svg_file}", shell=True, capture_output=True)


def dotbracket_to_basepairs(dotbracket):
    # Convert the dotbracket notation to base pairs
    '''
    dotbracket: the dotbracket notation
    return: the base pairs
    '''
    stack = []
    pairs = []
    for i, c in enumerate(dotbracket, start=1):
        if c in "([{<AB":
            stack.append(i)
        elif c in ")]}>a":
            if len(stack) == 0:
                return False
            j = stack.pop()
            pairs.append((j, i))
    if len(stack) > 0:
        return False
    return list(sorted(pairs))


def dot_to_tensor(dotbracket):
    # Convert the dotbracket notation to a tensor
    '''
    dotbracket: the dotbracket notation
    seq_len: the length of the sequence
    return: the tensor representing the dotbracket notation
    '''
    tensor = torch.zeros((len(dotbracket), len(dotbracket)), dtype=torch.float)

    pairs = dotbracket_to_basepairs(dotbracket)
    if pairs:
        for i, j in pairs:
            tensor[i - 1, j - 1] = 1
            tensor[j - 1, i - 1] = 1
    return tensor

def basepairs_to_tensor(base_pairs, seq_len):
    # Convert the base pairs to a tensor
    '''
    base_pairs: the base pairs
    seq_len: the length of the sequence
    return: the tensor representing the base pairs
    '''
    tensor = torch.zeros((seq_len, seq_len), dtype=torch.float)
    for i, j in base_pairs:
        tensor[i - 1, j - 1] = 1
        tensor[j - 1, i - 1] = 1
    return tensor

def pair_strength(base_pair):
    # Calculate the strength of a base pair
    '''
    base_pair: the base pair
    return: the strength of the base pair (1, 2, or 3), 0 if the base pair is invalid
    '''
    nu_dict = {
            "R": ["G","A"],
            "Y": ["C","U"],
            "K": ["G","U"],
            "M": ["A","C"],
            "S": ["G","C"],
            "W": ["A","U"],
            "B": ["G","U","C"],
            "D": ["G","A","U"],
            "H": ["A","C","U"],
            "V": ["G","C","A"],
            "N": ["A","G","C","U"]
        }
    
    if base_pair[0] in nu_dict and base_pair[1] in nu_dict[base_pair[0]]:
        if "G" in nu_dict[base_pair[0]] and "C" in nu_dict[base_pair[1]] or "C" in nu_dict[base_pair[0]] and "G" in nu_dict[base_pair[1]]:
            return 3
        elif "A" in nu_dict[base_pair[0]] and "U" in nu_dict[base_pair[1]] or "U" in nu_dict[base_pair[0]] and "A" in nu_dict[base_pair[1]]:
            return 2
        elif "G" in nu_dict[base_pair[0]] and "U" in nu_dict[base_pair[1]] or "U" in nu_dict[base_pair[0]] and "G" in nu_dict[base_pair[1]]:
            return 1
    else:
        if base_pair[0] == "G" and base_pair[1] == "C" or base_pair[0] == "C" and base_pair[1] == "G":
            return 3
        elif base_pair[0] == "A" and base_pair[1] == "U" or base_pair[0] == "U" and base_pair[1] == "A":
            return 2
        elif base_pair[0] == "G" and base_pair[1] == "U" or base_pair[0] == "U" and base_pair[1] == "G":
            return 1
    
    return 0

def probability_matrix(sequence):
    # Calculate the probability matrix
    '''
    base_pairs: the base pairs
    seq_len: the length of the sequence
    return: the probability matrix
    '''
    prob_matrix = np.zeros((len(sequence), len(sequence)), dtype= np.float32)

    base_pairs = np.array(np.meshgrid(len(sequence), len(sequence))).T.reshape(-1, 2)
    base_pairs = base_pairs[np.abs(base_pairs[:, 0] - base_pairs[:, 1]) > 3, :]

    for i, j in base_pairs:
        coefficient = 0
        for add in range(40):
            if (i - add >= 0) and (j + add < len(sequence)):
                score = pair_strength((sequence[i - add], sequence[j + add]))
                if score == 0:
                    break
                else:
                    coefficient += score * np.exp(-0.5 * (add**2))
            else:
                break
        if coefficient > 0:
            for add in range(1, 30):
                if (i + add < len(sequence)) and (j - add >= 0):
                    score = pair_strength((sequence[i + add], sequence[j - add]))
                    if score == 0:
                        break
                    else:
                        coefficient += score * np.exp(-0.5 * (add**2))
                else:
                    break

        prob_matrix[i, j] = coefficient

    return torch.tensor(prob_matrix)

def mat2bp(x):
    # Convert the matrix to base pairs
    '''
    matrix: the matrix
    th: the threshold
    return: the base pairs
    '''
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()
                
            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]   
            
    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
 
    return pairs_ind