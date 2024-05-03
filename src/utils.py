import torch
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from grakle import Graph
from grakle.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath


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

