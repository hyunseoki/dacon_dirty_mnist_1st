1. ~~dropout 30 -> 50~~ (2/17) -> lb 0.018 성능개선
1. ~~lr 1e-4 -> 1e-3 로 해볼 것~~ (2/18) -> lb0.019 개선
1. ~~bilateral filter 다시 해봐야함 (2/19)~~ -> history 9, 10번 비교하면 안쓴게 더 좋아보임
1. kfold ensemble (2/19~ ~~21~~ 23)
1. sigmoid + loss 바꿔서도 해봐야 함 torch.nn.BCEWithLogitsLoss(), torch.nn.MultiLabelSoftMarginLoss, torch.nn.BCELoss(),  https://discuss.pytorch.org/t/what-is-the-difference-between-bcewithlogitsloss-and-multilabelsoftmarginloss/14944/11 참고 (2/22)
1. 새로운 augmentation 테스트 성능 잘 나오는지 확인해봐야 함 (2/23)
1. scheduler 바꿔서도 해봐야 함 (warmup이 대새인 것 같다. Detectron2에서는 warmpup이 default값으로 셋팅되어 있음, https://pypi.org/project/pytorch-warmup/) (2/23)
1. regulation
