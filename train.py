import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import util
from model import MultiLabelEfficientNet
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/dirty_mnist_2nd/")
    parser.add_argument('--list_path', type=str, default="./data/dirty_mnist_2nd_answer.csv")
    parser.add_argument('--dataset_ratio', type=float, default=0.7)

    parser.add_argument('--model', type=str, default='efficientnet-b0')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patient', type=int, default=8)

    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()

    print('=' * 50)
    print('[info msg] arguments\n')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)
    
    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.list_path), 'wrong path'
    if (args.resume):
        assert os.path.isfile(args.resume), 'wrong path'

    util.seed_everything(777)

    data_set = util.DatasetMNIST(
        image_folder=args.image_path,
        label=args.list_path,
        transforms=util.mnist_transforms['train']
    )

    train_set_nb = int(len(data_set) * args.dataset_ratio)
    valid_set_nb = len(data_set) - train_set_nb
    train_set, val_set = torch.utils.data.random_split(data_set, [train_set_nb, valid_set_nb])

    train_data_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            shuffle = True,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size = args.batch_size,
            shuffle = False,
        )

    print('[info msg] {} model is created\n'.format(args.model))
    model = MultiLabelEfficientNet(args.model)
    print('=' * 50)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=2,
        factor=0.5,
        verbose=True
        )

    train_error = []
    valid_error = []

    best_loss = float("inf")
    best_loss_pos = None

    patient = 0
    patient_limit = args.patient

    date_time = datetime.now().strftime("%m%d%H%M")
    SAVE_DIR = os.path.join('./save', date_time)

    if(args.resume):
        model.load_state_dict(torch.load(args.resume))
        print('[info msg] pre-trained weight is loaded !!\n')        
        print(args.resume)
        print('=' * 50)

    print('[info msg] training start !!\n')

    startTime = datetime.now()

    if args.device is 'cuda' and torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    for epoch in range(args.epochs):        
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        train_loss = util.train(
            train_loader=train_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            optimizer=optimizer,
            )
        train_error.append(train_loss)

        valid_loss = util.validate(
            valid_loader=valid_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            scheduler=scheduler,
            )
        valid_error.append(valid_loss)        

        is_best = best_loss > valid_loss

        if is_best:
            patient = 0
            best_loss = valid_loss
            
            Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model_best.pth.tar'))
            print('MODEL SAVED!')
            
        else:
            patient += 1
            if patient > patient_limit - 1:
                print('=======' * 10)
                print("[Info message] Early stopper is activated")
                break

    elapsed_time = datetime.now() - startTime

    train_error = np.array(train_error)
    valid_error = np.array(valid_error)
    best_loss_pos = np.argmin(valid_error)
    
    print('=' * 50)
    print('[info msg] training is done\n')
    print("Time taken: {}".format(elapsed_time))
    print("best loss is {} at epoch : {}".format(best_loss, best_loss_pos))

    print('=' * 50)
    print('[info msg] {} model weight and log is save to {}\n'.format(args.model, SAVE_DIR))

    with open(os.path.join(SAVE_DIR, 'log.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value))            

        f.write('\n')
        f.write('total ecpochs : {}\n'.format(str(train_error.shape[0])))
        f.write('time taken : {}\n'.format(str(elapsed_time)))
        f.write('best_train_loss at {} epoch : {}\n'.format(np.argmin(train_error), np.min(train_error)))
        f.write('best_valid_loss at {} epoch : {}\n'.format(np.argmin(valid_error), np.min(valid_error)))

    plt.plot(train_error, label='train loss')
    plt.plot(valid_error, 'o', label='valid loss')
    plt.axvline(x=best_loss_pos, color='r', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'history.png'))
    plt.show()


if __name__ == '__main__':
    main()