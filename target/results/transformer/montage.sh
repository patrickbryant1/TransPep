for i in {0..4}
  do
    montage pairplot_tp_$i.png loss_tp_$i.png -tile 2x1 -geometry +2+2 pair_loss_$i.png
  done

montage pair_loss_0.png pair_loss_1.png pair_loss_2.png pair_loss_3.png pair_loss_4.png -tile 1x5 -geometry +2+2 all_pair_loss.png
