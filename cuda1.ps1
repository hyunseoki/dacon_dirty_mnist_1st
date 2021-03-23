$cmd0 = "python train_kfold.py --device cuda:1 --kfold_idx 4 --comment imagenet+new_augmentation+dropoout50+lr0.001+kfold4"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python train_kfold.py --device cuda:1 --kfold_idx 5 --comment imagenet+new_augmentation+dropoout50+lr0.001+kfold5"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0