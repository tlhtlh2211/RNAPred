import torch

# This class is used to embed the sequence of nucleotides into a tensor

class Embeds:

    # The class has the following methods:
    '''
    __init__(self): initializes the class
    embedSeq(self, seq, pads = "-"): embeds the sequence into a tensor
    '''

    def __init__(self):
        self.pads = "-"
        self.VOCAB = ["A","U","G","X"]
        self.size = 4
        # Mapping of nucleotide symbols to the corresponding nucleotides:
        # R:	Guanine / Adenine (purine)
        # Y:	Cytosine / Uracil (pyrimidine)
        # K:	Guanine / Uracil
        # M:	Adenine / Cytosine
        # S:	Guanine / Cytosine
        # W:	Adenine / Uracil
        # B:	Guanine / Uracil / Cytosine
        # D:	Guanine / Adenine / Uracil
        # H:	Adenine / Cytosine / Uracil
        # V:	Guanine / Cytosine / Adenine
        # N:	Adenine / Guanine / Cytosine / Uracil
        self.nu_dict = {
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
    
    def embedSeq(self, seq, pads = "-"):
        # Embeds the sequence into a tensor
        '''
        seq: the sequence to embed
        pads: the character used to pad the sequence
        return: the tensor embedding the sequence
        '''
        seq = seq.upper().replace("T", "U") # Convert to uppercase and replace T with U (RNA -> DNA)

        embed = torch.zeros((len(seq), self.size), dtype=torch.float) # Initialize the tensor

        for i, n in enumerate(seq):
            if n == pads: 
                continue
            elif n in self.VOCAB:
                embed[i, self.VOCAB.index(n)] = 1
            elif n in self.nu_dict: 
                for nu in self.nu_dict[n]:
                    embed[i, self.VOCAB.index(nu)] = 1 / len(self.nu_dict[n]) # Divide by the number of nucleotides in the group
            else:
                raise ValueError("Invalid nucleotide: " + n)
        
        return embed

