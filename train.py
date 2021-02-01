import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import util
from model import MultiLabelEfficientNet
import matplotlib.pyplot as plt
import torch


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"    

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/dirty_mnist/")
    parser.add_argument('--list_path', type=str, default="./data/dirty_mnist_answer.csv")
    parser.add_argument('--dataset_ratio', type=float, default=0.7)

    parser.add_argument('--model', type=str, default='efficientnet-b6')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patient', type=int, default=15)

    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    print('=' * 50)
    print('[info msg] arguments\n')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)
    
    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.list_path), 'wrong path'

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

    date_time = datetime.now().strftime("%m%d%H%M")
    SAVE_DIR = os.path.join('./save', date_time)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    train_error = []
    valid_error = []

    best_loss = float("inf")
    best_loss_pos = None

    patient = 0
    patient_limit = args.patient

    startTime = datetime.now()
    for epoch in range(args.epochs):        
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        train_loss = util.train(
            train_loader=train_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            optimizer=optimizer,
            scheduler=None,
            )
        train_error.append(train_loss)

        valid_loss = util.validate(
            valid_loader=valid_data_loader,
            model=model,
            loss_func=criterion,
            device=args.device,
            )
        valid_error.append(valid_loss)        

        is_best = best_loss > valid_loss

        if is_best:
            patient = 0
            best_loss = valid_loss
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
    
    print('[info msg] {} training is done\n')
    print("Time taken:", elapsed_time)
    print("best loss is {} at epoch : {}".format(best_loss, best_loss_pos))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model_best.pth.tar'))
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model_best.pth.tar')))

    print('[info msg] {} model weight and log is save to {}\n'.format(SAVE_DIR))

    with open(os.path.join(SAVE_DIR, 'log.txt'), 'w') as f:
        f.write('model : {}\n'.format(args.model))
        f.write('time taken : {}\n'.format(str(elapsed_time)))
        f.write('best_train_loss at {} epoch : {}\n'.format(np.argmin(train_error), np.min(train_error)))
        f.write('best_valid_loss at {} epoch : {}\n'.format(np.argmin(train_error), np.min(valid_error)))

    plt.plot(train_error, label='train loss')
    plt.plot(valid_error, 'o', label='valid loss')
    plt.axvline(x=best_loss_pos, color='r', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'history.png'))
    plt.show()


if __name__ == '__main__':
    main()