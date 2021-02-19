$cmd0 = "python train_kfold.py --device cuda:1 --kfold_idx 3 --comment imagenet+augmentation+dropoout50+lr0.001+kfold3"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd1 = "python train_kfold.py --device cuda:1 --kfold_idx 4 --comment imagenet+augmentation+dropoout50+lr0.001+kfold4"
$host.UI.RawUI.WindowTitle = $cmd1
Invoke-Expression -Command $cmd1

$cmd2 = "python train.py --device cuda:1 --lr 0.05 --comment imagenet+augmentation+dropoout50+lr0.05"
$host.UI.RawUI.WindowTitle = $cmd2
Invoke-Expression -Command $cmd2