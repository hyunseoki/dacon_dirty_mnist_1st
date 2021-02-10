import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import util
from efficientnet_pytorch import EfficientNet

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/dirty_mnist_2nd/")
    parser.add_argument('--label_path', type=str, default="./data/dirty_mnist_2nd_answer.csv")
    parser.add_argument('--dataset_ratio', type=float, default=0.7)

    parser.add_argument('--model', type=str, default='efficientnet-b7')
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
    assert os.path.isfile(args.label_path), 'wrong path'
    if (args.resume):
        assert os.path.isfile(args.resume), 'wrong path'

    util.seed_everything(777)

    data_set = pd.read_csv(args.label_path)
    train_set_nb = int(len(data_set) * args.dataset_ratio)
    # valid_set_nb = len(data_set) - train_set_nb

    train_set = util.DatasetMNIST(
        image_folder=args.image_path,
        label_df=data_set[:train_set_nb],
        transforms=util.mnist_transforms['train']
    )
    
    valid_set = util.DatasetMNIST(
        image_folder=args.image_path,
        label_df=data_set[train_set_nb:],
        transforms=util.mnist_transforms['valid']
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            shuffle = True,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size = args.batch_size,
            shuffle = False,
        )


    model = None
    
    if(args.resume):
        model = EfficientNet.from_name(args.model, in_channels=1, num_classes=26, dropout_rate=0.3)
        model.load_state_dict(torch.load(args.resume))
        print('[info msg] pre-trained weight is loaded !!\n')        
        print(args.resume)
        print('=' * 50)

    else:
        print('[info msg] {} model is created\n'.format(args.model))
        model = EfficientNet.from_pretrained(args.model, in_channels=1, num_classes=26, dropout_rate=0.3)
        print('=' * 50)

    if args.device == 'cuda' and torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)
 
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=2,
        factor=0.5,
        verbose=True
        )

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    best_loss = float("inf")

    patient = 0

    date_time = datetime.now().strftime("%m%d%H%M")
    SAVE_DIR = os.path.join('./save', date_time)

    print('[info msg] training start !!\n')
    startTime = datetime.now()
    for epoch in range(args.epochs):        
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        train_epoch_loss, train_epoch_acc = util.train(
            train_loader=train_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            optimizer=optimizer,
            )
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        valid_epoch_loss, valid_epoch_acc = util.validate(
            valid_loader=valid_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            scheduler=scheduler,
            )
        valid_loss.append(valid_epoch_loss)        
        valid_acc.append(valid_epoch_acc)

        if best_loss > valid_epoch_loss:
            patient = 0
            best_loss = valid_epoch_loss

            Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model_best.pth.tar'))
            print('MODEL IS SAVED TO {}!!!'.format(date_time))
            
        else:
            patient += 1
            if patient > args.patient - 1:
                print('=======' * 10)
                print("[Info message] Early stopper is activated")
                break

    elapsed_time = datetime.now() - startTime

    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)

    best_loss_pos = np.argmin(valid_loss)
    
    print('=' * 50)
    print('[info msg] training is done\n')
    print("Time taken: {}".format(elapsed_time))
    print("best loss is {} w/ acc {} at epoch : {}".format(best_loss, valid_acc[best_loss_pos], best_loss_pos))    

    print('=' * 50)
    print('[info msg] {} model weight and log is save to {}\n'.format(args.model, SAVE_DIR))

    with open(os.path.join(SAVE_DIR, 'log.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value))            

        f.write('\n')
        f.write('total ecpochs : {}\n'.format(str(train_loss.shape[0])))
        f.write('time taken : {}\n'.format(str(elapsed_time)))
        f.write('best_train_loss {} w/ acc {} at epoch : {}\n'.format(np.min(train_loss), train_acc[np.argmin(train_loss)], np.argmin(train_loss)))
        f.write('best_valid_loss {} w/ acc {} at epoch : {}\n'.format(np.min(valid_loss), valid_acc[np.argmin(valid_loss)], np.argmin(valid_loss)))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, 'o', label='valid loss')
    plt.axvline(x=best_loss_pos, color='r', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(valid_acc, 'o', label='valid acc')
    plt.axvline(x=best_loss_pos, color='r', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'history.png'))


if __name__ == '__main__':
    main()