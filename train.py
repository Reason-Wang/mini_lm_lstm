##
import os
import torch
import torch.nn as nn
import math
import torch.optim as optim
import give_valid_test
from model.lstm import TextLSTM
from data.process import make_batch
from data.process import make_dict
import argparse

def train_LSTMlm(device):
    model = TextLSTM(n_class=n_class, emb_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(args.epochs):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(args.data_dir, word2number_dict, args.n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % args.epochs_save == 0:
            torch.save(model, f'{args.ckpt_dir}/LSTMlm_model_{args.num_layers}_layer_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(args.data_dir, word2number_dict, args.n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_step",
        default=5,
        type=int,
        help="number of cells(number of step)"
    )

    parser.add_argument(
        "--hidden_size",
        default=128,
        type=int,
        help="Size of hidden states"
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size of data"
    )

    parser.add_argument(
        "--learning_rate",
        default=0.0005,
        type=float,
        help="Learning rate of optimizer"
    )

    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="Training epochs"
    )

    parser.add_argument(
        "--embed_size",
        default=256,
        type=int,
        help="Size of token embedding"
    )

    parser.add_argument(
        "--epochs_save",
        default=5,
        type=int,
        help="Number of epochs to save model"
    )

    parser.add_argument(
        '--data_dir',
        default='data/dataset',
        type=str,
        help="Directory of dataset"
    )

    parser.add_argument(
        "--num_layers",
        default=1,
        type=int,
        help="Number of layers of LSTM model"
    )

    parser.add_argument(
        "--ckpt_dir",
        default="model/ckpt",
        type=str,
        help="Directory to save checkpoint"
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_arguments()
    # n_step = 5  # number of cells(= number of Step)
    # n_hidden = 128  # number of hidden units in one cell
    # batch_size = 128  # batch size
    # learn_rate = 0.0005
    # all_epoch = 5  # the all epoch for training
    # emb_size = 256  # embeding size
    # save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    # data_root = 'data/dataset'

    train_path = os.path.join(args.data_dir, 'train.txt')  # the path of train dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("print parameter ......")
    print("n_step:", args.n_step)
    print("n_hidden:", args.hidden_size)
    print("batch_size:", args.batch_size)
    print("learn_rate:", args.learning_rate)
    print("all_epoch:", args.epochs)
    print("emb_size:", args.embed_size)
    print("save_checkpoint_epoch:", args.epochs_save)
    print("train_data:", args.data_dir)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)
    print(len(word2number_dict), len(number2word_dict))
    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, args.batch_size, args.n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, args.batch_size, args.n_step)
    all_target_batch = all_target_batch.reshape(-1, args.batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm(device)

    print("\nTest the LSTMLM……………………")
    select_model_path = f"{args.ckpt_dir}/LSTMlm_model_{args.num_layers}_layer_epoch5.ckpt"
    test_LSTMlm(select_model_path)


