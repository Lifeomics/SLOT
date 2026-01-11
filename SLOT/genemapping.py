import pandas as pd
import numpy as np


def genename_transfer(raw_genes, mapping_dict_path):
    """
    Transfer gene names from raw_genes to the gene names in the reference gene set.
    
    Parameters
    ----------
    raw_genes: list
        A list of gene names.
    mapping_dict_path: str
        Path to the mapping dictionary.
        
    Returns
    -------
    genes: list
        A list of gene names after conversion.
    """
    
    # Load the mapping dictionary from the provided file path
    mapping_df = pd.read_csv(mapping_dict_path, sep='\t', header=None)
    
    # Create a dictionary to map synonyms to gene symbols
    synonyms_to_symbol = {}
    for _, row in mapping_df.iterrows():
        # Check if "gene synonyms" column is not NaN and split only if it's a valid string
        if isinstance(row[4], str):  # Ensure that the value is a string
            synonyms = row[4].split("|")  # Split synonyms by "|" (index 4 refers to "gene synonyms" column)
            for synonym in synonyms:
                synonyms_to_symbol[synonym] = row[1]  # Map synonym to gene symbol (index 1 refers to "gene symbol")
        else:
            # Handle cases where "gene synonyms" is missing (NaN or other invalid data)
            continue
    
    # Convert Gene Synonyms to Gene Symbols
    converted_gene_symbols = []
    for synonym in raw_genes:
        # If the gene name is already a Gene Symbol, do not convert
        if synonym in synonyms_to_symbol:
            converted_gene_symbols.append(synonyms_to_symbol[synonym])
        else:
            converted_gene_symbols.append(synonym)
    
    return converted_gene_symbols