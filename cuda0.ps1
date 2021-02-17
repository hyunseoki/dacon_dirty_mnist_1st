$cmd0 = "python train.py --device cuda:0 --model efficientnet-b8 --comment imagenet+augmentation+dropoout50"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd1 = "python train.py --device cuda:0 --lr 0.001 --model efficientnet-b8 --comment imagenet+augmentation"
$host.UI.RawUI.WindowTitle = $cmd1
Invoke-Expression -Command $cmd1