import pickle
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import basepairs_to_tensor
from embeds import Embeds
import json


# This class is used to load the data

class Data(Dataset):
    def __init__(self, path, cache, min_len = 0, max_len = 512, prediction = False, cannonical_mask = False, prior = "Probability Matrix"):
        # __init__ has following parameters:
        '''
        path: the path to the data
        min_len: the minimum length of the sequence
        max_len: the maximum length of the sequence
        prediction: whether the data is used for prediction (default: False, True if used for prediction)
        canonical_mask: whether to use a canonical mask (default: False)
        prior: the prior to use (none, default: "Probability Matrix", "Pairwise")
        '''

        self.min_len = min_len
        self.max_len = max_len
        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.embedding = Embeds()
        self.embedding_size = self.embedding.size
        self.prior = prior
        self.cannonical_mask = cannonical_mask
        self.cache = cache
        
        # Create the cache directory if it does not exist
        if cache is not None and not os.path.isdir(cache):
            os.mkdir(cache)

        # Load the data
        data = pd.read_csv(path)

        # Filter the data
        '''
        The data is filtered based on the existence of the columns 'sequence' and 'id' and, 
        if prediction is False, 'base_pairs' or 'dotbracket'
        '''
        if (prediction):
            assert (
                "sequence" in data.columns and "id" in data.columns
            ), "The data must have the columns 'sequence' and 'id' to make predictions"
        
        else:
            assert(
                "sequence" in data.columns and "id" in data.columns
                and ("base_pairs" in data.columns or "dotbracket" in data.columns)
            ), "The data must have the columns 'sequence', 'id', and 'base_pairs' or 'dotbracket' to train the model"

        # Filter the data based on the length of the sequence
        data["length_seq"] = data.sequence.str.len()
        data = data[(data.length_seq >= self.min_len) & (data.length_seq <= self.max_len)]

        # Add base_pairs from data to the class
        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [
                json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        idx : the index of the sequence
        return: the sequence and the base pairs
        '''
        seq = self.sequences[idx]
        id = self.ids[idx]

        if (self.cache is not None) and not os.path.isdir(f"{self.cache}/{id}.pk"):
            return pickle.load(open(f"{self.cache}/{id}.pk", "rb"))

        if self.base_pairs is not None:
            base_pairs = self.base_pairs[idx]
            matrix = basepairs_to_tensor(base_pairs, len(seq))
        
        sequence_embed = self.embedding.embedSeq(seq)

        if self.cannonical_mask:
            mask = self.cannonical_mask
        else:
            mask = None
        
        if self.prior == "Probability Matrix":
            prior = self.embedding.probability_matrix(seq)
        else:
            prior = None
        
        item = {"embedding": sequence_embed, "contact": matrix, "length": len(seq), "canonical_mask": mask,
                    "id":id, "sequence": seq, "interaction_prior": prior}

        if self.cache is not None:
            pickle.dump(item, open(f"{self.cache}/{id}.pk", "wb"))

        return item