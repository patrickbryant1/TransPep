# TransPep

Signal peptides are essential for protein sorting and processing. Evaluating signal peptides experimentally is difficult and prone to errors, therefore the exact cleavage sites are often misannotated. Here, we describe a novel explainable method to identify signal peptides and predict the cleavage site, with a performance similar to state-of-the art methods. We treat each amino acid sequence as a sentence and its annotation as a translation problem. We utilise attention neural networks in a transformer model using a simple one-hot representation of each amino acid, without including any evolutionary information. By analysing the encoder-decoder attention of the trained network, we are able to explain what information in the peptide is used to annotate the cleavage site. We find the most common signal peptide motifs and characteristics and confirm that the most informative amino acid sites vary greatly between kingdoms and signal peptide types as previous studies have shown. Our findings open up the possibility to gain biological insight using transformer neural networks on small sets of labelled information.


**Studying signal peptides with attention neural networks informs cleavage site predictions**
by Bryant et al. has been accepted to the Machine Learning for Structural Biology (MLSB) Workshop at NeurIPS 2021. 
