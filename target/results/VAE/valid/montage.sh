for i in 1 2 3 4
do
montage type_umap0_$i.png type_umap_seq0_$i.png -tile 2x1 -geometry +2+2 types0_$i.png
montage org_umap0_$i.png org_umap_seq0_$i.png -tile 2x1 -geometry +2+2 orgs0_$i.png
done

montage types0_1.png types0_2.png types0_3.png types0_4.png  -tile 1x4 -geometry +2+2 types0.png
montage orgs0_1.png orgs0_2.png orgs0_3.png orgs0_4.png  -tile 1x4 -geometry +2+2 orgs0.png
