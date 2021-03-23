$cmd0 = "python train_kfold.py --device cuda:0 --kfold_idx 1 --comment imagenet+new_augmentation+dropoout50+lr0.001+kfold1"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python train_kfold.py --device cuda:0 --kfold_idx 2 --comment imagenet+new_augmentation+dropoout50+lr0.001+kfold2"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0