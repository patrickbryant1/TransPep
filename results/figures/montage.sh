#Bench
# montage Sec_SPI_MCC.png Sec_SPI_Recall.png Sec_SPI_Precision.png -tile 1x3 -geometry +2+2 Sec_SPI_bench.png
# montage Sec_SPII_MCC.png Sec_SPII_Recall.png Sec_SPII_Precision.png -tile 1x3 -geometry +2+2 Sec_SPII_bench.png
# montage Tat_SPI_MCC.png Tat_SPI_Recall.png Tat_SPI_Precision.png -tile 1x3 -geometry +2+2 Tat_SPI_bench.png
# montage Sec_SPI_bench.png Sec_SPII_bench.png Tat_SPI_bench.png -tile 3x1 -geometry +2+2 all_bench.png
# montage legend.png all_bench.png -tile 1x2 -geometry +2+2 all_bench.png

#Probability cutoffs
#montage /home/patrick/results/protein_translation/attention/ARCHAEA/type_probs.png /home/patrick/results/protein_translation/attention/EUKARYA/type_probs.png /home/patrick/results/protein_translation/attention/NEGATIVE/type_probs.png /home/patrick/results/protein_translation/attention/POSITIVE/type_probs.png -tile 1x4 -geometry +2+2 type_probs.png
#CS Precision
# BASE=/home/patrick/results/protein_translation/attention/
# for TYPE in ARCHAEA NEGATIVE POSITIVE
#   do
#     montage $BASE'/'$TYPE'/'$TYPE'_CS_precision1.png'  $BASE'/'$TYPE'/'$TYPE'_CS_precision3.png' $BASE'/'$TYPE'/'$TYPE'_CS_precision2.png' -tile 3x1 -geometry +2+2 $TYPE'_CS_prec.png'
#   done
# montage ARCHAEA_CS_prec.png /home/patrick/results/protein_translation/attention/EUKARYA/EUKARYA_CS_precision1.png NEGATIVE_CS_prec.png POSITIVE_CS_prec.png -tile 1x4 -geometry +2+2 'CS_prec.png'
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
montage  logos3.png logos2.png -tile 2x1 -geometry +2+2 logos_2_3.png


#Montage attention matrices
# BASE=/home/patrick/results/protein_translation/attention/
# for TYPE in ARCHAEA NEGATIVE POSITIVE
#   do
#   for i in 1 2 3
#   do
#     montage $BASE'/'$TYPE'/'$TYPE'_enc_dec_attention_'$i'_TP_CS_area_bar.png' $BASE'/'$TYPE'/'$TYPE'_enc_dec_attention_'$i'_TP_CS_area.png' -tile 1x2 -geometry +2+2 $TYPE'_'$i'_attention_bar.png'
#   done
#   done
# #Eukarya
# TYPE=EUKARYA
# i=1
# montage $BASE'/'$TYPE'/'$TYPE'_enc_dec_attention_'$i'_TP_CS_area_bar.png' $BASE'/'$TYPE'/'$TYPE'_enc_dec_attention_'$i'_TP_CS_area.png' -tile 1x2 -geometry +2+2 $TYPE'_'$i'_attention_bar.png'
#
# #Together
# #Sec/SPI
# montage ARCHAEA_1_attention_bar.png EUKARYA_1_attention_bar.png NEGATIVE_1_attention_bar.png POSITIVE_1_attention_bar.png -tile 2x2 -geometry +2+2 matrices1.png
# #Tat/SPI
# montage ARCHAEA_2_attention_bar.png NEGATIVE_2_attention_bar.png POSITIVE_2_attention_bar.png -tile 1x3 -geometry +2+2 matrices2.png
# #Sec/SPII
# montage ARCHAEA_3_attention_bar.png NEGATIVE_3_attention_bar.png POSITIVE_3_attention_bar.png -tile 1x3 -geometry +2+2 matrices3.png
#2 and 3 together
#montage matrices3.png matrices2.png -tile 2x1 -geometry +2+2 matrices_2_3.png

#
#Sec/SPI
#montage $BASE'/ARCHAEA/ARCHAEA_enc_dec_attention_1_TP_CS_area.png' $BASE'/EUKARYA/EUKARYA_enc_dec_attention_1_TP_CS_area.png' $BASE'/NEGATIVE/NEGATIVE_enc_dec_attention_1_TP_CS_area.png' $BASE'/POSITIVE/POSITIVE_enc_dec_attention_1_TP_CS_area.png' -tile 2x2 -geometry +2+2 matrices1.png



#Montage the CS error
# BASE=/home/patrick/results/protein_translation/attention/
# montage $BASE/ARCHAEA/ARCHAEA_CS_diff_1.png $BASE/EUKARYA/EUKARYA_CS_diff_1.png $BASE/NEGATIVE/NEGATIVE_CS_diff_1.png $BASE/POSITIVE/POSITIVE_CS_diff_1.png -tile 4x1 -geometry +2+2 CS_diff_1.png
# montage $BASE/ARCHAEA/ARCHAEA_CS_diff_2.png $BASE/NEGATIVE/NEGATIVE_CS_diff_2.png $BASE/POSITIVE/POSITIVE_CS_diff_2.png -tile 3x1 -geometry +2+2 CS_diff_2.png
# montage $BASE/ARCHAEA/ARCHAEA_CS_diff_3.png $BASE/NEGATIVE/NEGATIVE_CS_diff_3.png $BASE/POSITIVE/POSITIVE_CS_diff_3.png -tile 3x1 -geometry +2+2 CS_diff_3.png
# montage CS_diff_1.png CS_diff_2.png CS_diff_3.png -tile 1x3 -geometry +2+2 CS_diff.png
