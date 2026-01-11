from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.go_enrichment import GOEnrichmentStudy
import pandas as pd



def read_gaf_by_gene_name(gaf_file, taxids=None):
    gene2gos = {}
    with open(gaf_file, 'r') as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 15:
                continue
            
            if taxids:
                taxon_info = parts[12]
                valid = False
                for t in taxids:
                    if f"taxon:{t}" in taxon_info:
                        valid = True
                        break
                if not valid:
                    continue
            
            gene_name = parts[2]  
            go_id = parts[4]      
            if gene_name not in gene2gos:
                gene2gos[gene_name] = set()
            gene2gos[gene_name].add(go_id)
    return gene2gos

def get_goea_filtered_dataframe(results, obodag, pval_threshold, namespace_filter=None, min_depth=2, gene2gos=None, gene_list=None):
    """
    Filter GO enrichment analysis results to retain only those with an FDR p-value <= pval_threshold,
    and optionally filter by GO namespace. Returns a DataFrame containing the columns:
    GO, Name, Namespace, p_uncorrected, p_fdr_bh, study_count, and Genes.
    
    Parameters:
      results: List of GO enrichment analysis result objects. Each result object should include the attributes:
               - GO: GO term ID.
               - name: GO term name.
               - p_uncorrected: Uncorrected p-value.
               - p_fdr_bh: FDR corrected p-value.
               - study_count: The number of genes in the study set associated with the GO term.
      obodag: GO DAG object loaded via GODag, used to obtain the namespace information for each GO term.
      pval_threshold: The filtering threshold. Only results with an FDR p-value <= pval_threshold are retained.
      namespace_filter: Optional filter for the GO namespace. Can be set to "BP" (biological_process), 
                        "MF" (molecular_function), or "CC" (cellular_component). If None, no namespace filtering is applied.
      min_depth: The minimum depth of the GO term to retain. Default is 2.
      gene2gos: Dictionary mapping genes to their associated GO terms.
    
    Returns:
      A pandas DataFrame containing the columns: GO, Name, Namespace, p_uncorrected, p_fdr_bh, study_count, and Genes.
    """
    records = []
    # Map shorthand namespace filter to the full namespace name, if provided.
    if namespace_filter is not None:
        mapping = {"BP": "biological_process", "MF": "molecular_function", "CC": "cellular_component"}
        ns_filter = mapping.get(namespace_filter, namespace_filter)
    else:
        ns_filter = None

    for res in results:
        # Filter based on p-value threshold and study_count > 0.
        if res.p_fdr_bh <= pval_threshold and res.study_count > 0:
            if res.GO in obodag and obodag[res.GO].depth < min_depth:
                continue
            # Retrieve the namespace for the GO term; if not found, mark as "NA"
            ns = obodag[res.GO].namespace if res.GO in obodag else "NA"
            # If a namespace filter is specified, only retain records that match the filter.
            if ns_filter is not None and ns != ns_filter:
                continue
            # Retrieve the list of genes associated with the GO term, filtered by gene_list
            genes = [gene for gene, gos in gene2gos.items() if res.GO in gos and gene in gene_list]
            record = {
                "GO": res.GO,
                "Name": res.name,
                "Namespace": ns,
                "p_uncorrected": res.p_uncorrected,
                "p_fdr_bh": res.p_fdr_bh,
                "study_count": res.study_count,
                "Genes": genes
            }
            records.append(record)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values(by="study_count", ascending=False)
    return df

def run_go_enrichment_analysis(gene_list:list, background_genes:list, gene2gos, obodag, pval_threshold=0.05, namespace_filter="CC", min_depth=2):
    """
    Perform GO enrichment analysis and return a filtered DataFrame containing the results.

    Parameters:
        gene_list (list): List of genes to analyze in the GO enrichment study.
        background_genes (list): List of background genes used to perform the enrichment analysis.
        gene2gos (dict): Dictionary mapping genes to their associated GO terms.
        obodag (GODag): The GO ontology DAG object, used to retrieve namespace information for each GO term.
        pval_threshold (float): The threshold for FDR p-value to filter the results. Default is 0.05.
        namespace_filter (str): Optional filter for GO namespaces. Valid values are "BP" (biological_process), 
                                 "MF" (molecular_function), or "CC" (cellular_component). Default is "CC".
        min_depth (int): The minimum depth for GO terms to retain. Default is 2.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing the results of the GO enrichment analysis with the following columns:
                          "GO", "Name", "Namespace", "p_uncorrected", "p_fdr_bh", "study_count", and "Genes".
    """
    goea_obj = GOEnrichmentStudy(
        background_genes,
        gene2gos,
        obodag,
        methods=['fdr_bh'],
        min_overlap=3
    )

    results = goea_obj.run_study(gene_list)
    
    # Pass gene2gos to get_goea_filtered_dataframe
    return get_goea_filtered_dataframe(results, obodag, pval_threshold, namespace_filter, min_depth, gene2gos, gene_list)
