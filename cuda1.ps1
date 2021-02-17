$cmd0 = "python train.py --device cuda:1 --model efficientnet-b8 --comment imagenet+augmentation+moonfilter1"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0