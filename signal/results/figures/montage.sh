#LOGOS
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/attention_logo1.png /home/patrick/results/protein_translation/attention/EUKARYA/attention_logo1.png /home/patrick/results/protein_translation/attention/NEGATIVE/attention_logo1.png /home/patrick/results/protein_translation/attention/POSITIVE/attention_logo1.png -tile 2x2 -geometry +2+2 attention_logo1.png

#Probability cutoffs
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/type_probs.png /home/patrick/results/protein_translation/attention/EUKARYA/type_probs.png /home/patrick/results/protein_translation/attention/NEGATIVE/type_probs.png /home/patrick/results/protein_translation/attention/POSITIVE/type_probs.png -tile 1x4 -geometry +2+2 type_probs.png
#Precision
montage /home/patrick/results/protein_translation/attention/ARCHAEA/type_precisions.png /home/patrick/results/protein_translation/attention/EUKARYA/EUKARYA_type_precision1.png /home/patrick/results/protein_translation/attention/NEGATIVE/type_precisions.png /home/patrick/results/protein_translation/attention/POSITIVE/type_precisions.png -tile 1x4 -geometry +2+2 type_precisions.png
