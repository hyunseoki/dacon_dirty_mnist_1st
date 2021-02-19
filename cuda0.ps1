$cmd0 = "python train_kfold.py --device cuda:0 --kfold_idx 0 --comment imagenet+augmentation+dropoout50+lr0.001+kfold0"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd1 = "python train_kfold.py --device cuda:0 --kfold_idx 1 --comment imagenet+augmentation+dropoout50+lr0.001+kfold1"
$host.UI.RawUI.WindowTitle = $cmd1
Invoke-Expression -Command $cmd1

$cmd2 = "python train_kfold.py --device cuda:0 --kfold_idx 2 --comment imagenet+augmentation+dropoout50+lr0.001+kfold2"
$host.UI.RawUI.WindowTitle = $cmd2
Invoke-Expression -Command $cmd2