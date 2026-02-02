for i in 1 2 3 4
do
  python train.py --category airplane --part_label $i
done

#python train.py -t