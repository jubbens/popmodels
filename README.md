Code for the analyses in *Quantitative Evaluation of Nonlinear Methods for Population Structure Visualization and Inference* by Ubbens et al.

## Dependencies

Requires [SeqBreed](https://github.com/miguelperezenciso/SeqBreed). Other dependencies can be installed via conda using `conda env create -f env.yml`.

## Files

`parse_genotypes_csv.py` - Convert a csv file of genotypes to a binary file.  
`parse_genotypes_plink.py` - Convert a plink file of genotypes to a binary file.  
`parse_pedigree_matrix.py` - Convert a pedigree matrix represented in a csv file to a binary file.  
`evaluate_autoencoder.py` - Run the autoencoder model for given genotype binary, outputs plots and a distance matrix file.  
`evaluate_vae.py` - Run the VAE model for given genotype binary, outputs plots and a distance matrix file.  
`evaluate_contrastive.py` - Run the contrastive embedding learning model for given genotype binary, outputs plots and a distance matrix file.  
`evaluate_random_projection.py` - Run the random projection model for given genotype binary, outputs plots and a distance matrix file.  
`compare_pedigree_baselines.py` - Get results for PCA, t-SNE, UMAP, and MDS for given genotype and pedigree binaries.  
`compare_pedigree_custom.py` - Get results for an existing distance matrix for given genotype and pedigree binaries.  
`simpop_multi_runner.py` - Run the simulation experiments for Figures 1, 3, and 4.

