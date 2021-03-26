#Bench
# montage Sec_SPI_MCC.png Sec_SPI_Recall.png Sec_SPI_Precision.png -tile 1x3 -geometry +2+2 Sec_SPI_bench.png
# montage Sec_SPII_MCC.png Sec_SPII_Recall.png Sec_SPII_Precision.png -tile 1x3 -geometry +2+2 Sec_SPII_bench.png
# montage Tat_SPI_MCC.png Tat_SPI_Recall.png Tat_SPI_Precision.png -tile 1x3 -geometry +2+2 Tat_SPI_bench.png
# montage Sec_SPI_bench.png Sec_SPII_bench.png Tat_SPI_bench.png -tile 3x1 -geometry +2+2 all_bench.png
# montage legend.png all_bench.png -tile 1x2 -geometry +2+2 all_bench.png
#LOGOS
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/attention_logo1.png /home/patrick/results/protein_translation/attention/EUKARYA/attention_logo1.png /home/patrick/results/protein_translation/attention/NEGATIVE/attention_logo1.png /home/patrick/results/protein_translation/attention/POSITIVE/attention_logo1.png -tile 2x2 -geometry +2+2 attention_logo1.png

# montage /home/patrick/results/protein_translation/attention/ARCHAEA/attention_logo2.png /home/patrick/results/protein_translation/attention/NEGATIVE/attention_logo2.png /home/patrick/results/protein_translation/attention/POSITIVE/attention_logo2.png -tile 1x3 -geometry +2+2 attention_logo2.png
# montage /home/patrick/results/protein_translation/attention/ARCHAEA/attention_logo3.png /home/patrick/results/protein_translation/attention/NEGATIVE/attention_logo3.png /home/patrick/results/protein_translation/attention/POSITIVE/attention_logo3.png -tile 1x3 -geometry +2+2 attention_logo3.png
# montage attention_logo2.png  attention_logo3.png -tile 2x3 -geometry +2+2  attention_logo2_3.png

#Probability cutoffs
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/type_probs.png /home/patrick/results/protein_translation/attention/EUKARYA/type_probs.png /home/patrick/results/protein_translation/attention/NEGATIVE/type_probs.png /home/patrick/results/protein_translation/attention/POSITIVE/type_probs.png -tile 1x4 -geometry +2+2 type_probs.png
#Precision
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/type_precisions.png /home/patrick/results/protein_translation/attention/EUKARYA/EUKARYA_type_precision1.png /home/patrick/results/protein_translation/attention/NEGATIVE/type_precisions.png /home/patrick/results/protein_translation/attention/POSITIVE/type_precisions.png -tile 1x4 -geometry +2+2 type_precisions.png


#TP total attention distribution
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/TP_attention_type.png /home/patrick/results/protein_translation/attention/EUKARYA/EUKARYA_enc_dec_attention_1_TP.png /home/patrick/results/protein_translation/attention/NEGATIVE/TP_attention_type.png /home/patrick/results/protein_translation/attention/POSITIVE/TP_attention_type.png -tile 1x4 -geometry +2+2 attention_TP.png


#montage sequence frequency and attention Logos
BASE=/home/patrick/results/protein_translation/attention/
for TYPE in ARCHAEA NEGATIVE POSITIVE
  do
  for i in 1 2 3
  do
    montage $BASE'/'$TYPE'/'$TYPE'_aa_enc_dec_attention_logo_'$i'.png' $BASE'/'$TYPE'/'$TYPE'_aa_seq_logo_'$i'.png' -tile 1x2 -geometry +2+2 $TYPE'_'$i'_attention_freq.png'
  done
  done

#Eukarya
TYPE=EUKARYA
i=1
montage $BASE'/'$TYPE'/'$TYPE'_aa_enc_dec_attention_logo_'$i'.png' $BASE'/'$TYPE'/'$TYPE'_aa_seq_logo_'$i'.png' -tile 1x2 -geometry +2+2 $TYPE'_'$i'_attention_freq.png'

#Together
#Sec/SPI
montage ARCHAEA_1_attention_freq.png EUKARYA_1_attention_freq.png NEGATIVE_1_attention_freq.png POSITIVE_1_attention_freq.png -tile 2x2 -geometry +2+2 logos1.png
#Tat/SPI
montage ARCHAEA_2_attention_freq.png NEGATIVE_2_attention_freq.png POSITIVE_2_attention_freq.png -tile 1x3 -geometry +2+2 logos2.png
#Sec/SPII
montage ARCHAEA_3_attention_freq.png NEGATIVE_3_attention_freq.png POSITIVE_3_attention_freq.png -tile 1x3 -geometry +2+2 logos3.png
#2 and 3 together
montage logos2.png logos3.png -tile 2x1 -geometry +2+2 logos_2_3.png
