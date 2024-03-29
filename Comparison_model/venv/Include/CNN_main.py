import torch
import random
import numpy as np
import torch.nn as nn
import visdom
import os
import time
import matplotlib.pyplot as plt
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.my_utils import t_sne, umap_fun2
from models.CNN_model import CNN
from proto_data_utils.train_utils import weights_init, weights_init2, set_seed
from proto_data_utils.plot_utils import tSNE_fun

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vis = visdom.Visdom(env='yancy_env')
generator = data_generate()
CHECK_EPOCH = 10
Tr_EPOCHS = 200
Te_EPOCHS = 1
DIM = 2048  # 2048
# DIM = 1024
Load = [3, 0]


# ------------------------ DASMN paper: Tools---------------
def save_model(model, filename):
    if os.path.exists(filename):
        filename += '(1)'
    torch.save(model.state_dict(), filename)
    print('This model is saved at [%s]' % filename)


def check_creat_new(path):
    if os.path.exists(path):
        split_f = os.path.split(path)
        new_f = os.path.join(split_f[0], split_f[1][:-4] + '(1).svg')
        new_f = check_creat_new(new_f)  # in case that the new file exist
    else:
        new_f = path
    return new_f


def plot_adaptation(x_s, x_t, shot):
    x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
    print('CW2SQ labels used for t-sne!')
    labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
    tSNE_fun(x.cpu().detach().numpy(), shot=shot, name=None, labels=labels, n_dim=2)


# -----------------------------------------


def compute_acc(out, y):
    """
    :param out: the result of classifier
    :param y: labels
    :return: accuracy
    """
    prob = nn.functional.log_softmax(out, dim=-1)
    pre = torch.max(prob, dim=1)[1]
    # print('y_label:\n', y.cpu().numpy())
    # print('predicted:\n', pre.cpu().numpy())
    acc = torch.eq(pre, y).float().mean().cpu().item()
    return acc


def train(net, save_path, train_x, train_y, tar_x, tar_y,
          ls_threshold=1e-5, n_way=3, shot=3):
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    print('Adam-optimizer!')
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
    loss = nn.CrossEntropyLoss()
    # input(y_, y) y_:[n, Nc] y:[n] and y_ should not be normalized such as soft-max.

    n_examples = train_x.shape[1]

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('train_x Shape:', train_x.shape)
    print('train_y Shape:', train_y.shape)
    print("---------------------Training----------------------\n")
    counter = 0
    n_episodes = n_examples // shot
    avg_ls = torch.zeros([n_episodes]).to(device)
    n_epochs = Tr_EPOCHS
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        count = 0
        for epi in range(n_episodes):
            x, y = train_x[:, count:count + shot], train_y[:, count:count + shot]
            t_x, t_y = tar_x[:, count:count + shot], tar_y[:, count:count + shot]
            x, y = x.to(device), y.contiguous().view(-1).to(device)
            count += shot

            x, y = x.to(device), y.contiguous().view(-1).to(device)
            t_x, t_y = t_x.to(device), t_y.to(device).view(-1).to(device)

            outputs, _ = net.forward(x)
            losses = loss(outputs, y)
            avg_ls[epi] = losses
            acc = compute_acc(outputs, y)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                t_out, _ = net.forward(t_x)
            net.train()

            t_loss = loss(t_out, t_y)
            t_acc = compute_acc(t_out, t_y)

            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, losses.cpu().item(), acc))
            if (epi + 1) % 2 == 0:
                vis.line(Y=[[losses.cpu().item(), t_loss.cpu().item()]], X=[counter],
                         update=None if counter == 0 else 'append', win='CNN_Loss',
                         opts=dict(legend=['src_loss', 'tar_loss'], title='CNN_Loss'))
                vis.line(Y=[[acc, t_acc]], X=[counter],
                         update=None if counter == 0 else 'append', win='CNN_Acc',
                         opts=dict(legend=['src_acc', 'tar_acc'], title='CNN_Acc'))
                counter += 1
        scheduler.step()
        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_train_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))

        # if ep + 1 >= CHECK_EPOCH and (ep + 1) % 10 == 0:
        #     order = input("Shall we stop training now? (Epoch {}) Y/N\n".format(ep + 1))
        #     order = order is 'Y' or order is 'y'
        # else:
        #     order = False
        #
        # if ls_ <= ls_threshold and order:
        #     print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
        #     break
        # elif order:
        #     print('Stop manually.\n')
        #     break
        if ep + 1 >= CHECK_EPOCH and (ep + 1) % 5 == 0:
            flag = input("Shall we stop the training? Y/N\n")
            if flag == 'y' or flag == 'Y':
                print('Training stops!(manually)')
                new_path = os.path.join(save_path, f"final_epoch{ep + 1}")
                save_model(net, new_path)
                break
            else:
                flag = input(f"Save model at epoch {ep + 1}? Y/N\n")
                if flag == 'y' or flag == 'Y':
                    child_path = os.path.join(save_path, f"epoch{ep + 1}")
                    save_model(net, child_path)

    print('train finished!')


def test(save_path, save_fig_path, test_x, test_y, src_x=None, test_scenario='test',
         n_way=3, shot=3, eval=False):
    net = CNN(nc=n_way, DIM=DIM).to(device)
    net.load_state_dict(torch.load(save_path))
    print("Load the model Successfully！\n%s" % save_path)
    net = net.eval() if eval else net.train()
    print('Model.eval() is:', not net.training)
    loss = nn.CrossEntropyLoss()

    n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert test_x.shape[1] == test_x.shape[1]
    assert n_way == n_class

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    if src_x is not None:
        print('src_data set Shape:', src_x.shape)
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    n_episodes = n_examples // shot
    counter = 0
    n_epochs = Te_EPOCHS
    avg_time = []
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        time_ep = []
        count = 0
        for epi in range(n_episodes):
            x, y = test_x[:, count:count + shot], test_y[:, count:count + shot]
            x, y = x.to(device), y.reshape(-1).to(device)
            count += shot
            if src_x is not None:
                selected = torch.randperm(src_x.shape[1])[:shot]
                s_x = src_x[:, selected].to(device)
            t0 = time.time()
            with torch.no_grad():
                outputs, f_t = net.forward(x)
                if src_x is not None:
                    _, f_s = net.forward(s_x)
            t1 = time.time()

            losses = loss(outputs, y)
            ls = losses.cpu().item()
            ac = compute_acc(outputs, y)
            time_ep.append(t1 - t0)
            avg_acc += ac
            avg_loss += ls

            vis.line(Y=[[ac, ls]], X=[counter],
                     update=None if counter == 0 else 'append', win=test_scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=test_scenario))
            counter += 1

            # if (epi + 1) == episodes:
            plot_adaptation(f_s, f_t, shot=shot)
            plt.show()
            order = input('Save fig? Y/N\n')
            if order == 'y' or order == 'Y':
                plot_adaptation(f_s, f_t, shot=shot)
                new_path = check_creat_new(save_fig_path)
                plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
                print('Save t-SNE.eps to \n', new_path)

        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        avg_time.append(np.sum(time_ep))
        print('[epoch {}/{}]\tavg_time: {:.4f}\tavg_loss: {:.8f}\tavg_acc: {:.8f}'.
              format(ep + 1, n_epochs, avg_time[-1], avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    # vis.text('Average Accuracy: {:.6f}  Average Loss:{:.6f}'.format(avg_acc_, avg_loss_), win='Test result')
    vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not net.training, avg_acc_),
             win='Eval:{} Test result'.format(not net.training))
    print('\n------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.4f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}'.format(avg_loss_))
    print('Average Test Time: {:.4f}\n'.format(np.mean(avg_time)))


def main(save_path, fig_path, ls_threshold=0.001, n_way=3, shot=3, split=10, ob_domain=False):
    print('%d GPU is available.' % torch.cuda.device_count())
    set_seed(0)
    net = CNN(nc=n_way, DIM=DIM).to(device)
    # net.apply(weights_init2)  # 教训：CW2SQ时，对CNN不推荐使用手动初始化

    # split = 50 if ob_domain else split

    # CW: NC, IF, OF, RoF
    # train_x, train_y, test_x, test_y = generator.Cs_4way(way=n_way, examples=50, split=split,
    #                                                      normalize=True, data_len=DIM,
    #                                                      label=True, shuffle=True)
    # SQ 7
    # train_x, train_y, test_x, test_y = generator.SQ_37way(way=n_way, examples=100, split=split,
    #                                                       shuffle=False, data_len=DIM,
    #                                                       normalize=True, label=True)
    # CW: NC, IF, OF, RoF
    # train_x, train_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split,
    #                                             normalize=True, data_len=DIM, SNR=None, label=True)
    # _, _, test_x, test_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0,
    #                                           normalize=True, data_len=DIM, SNR=None, label=True)

    # ======================
    # CW2SA
    # train_x, train_y, _, _ = generator.CW_cross(way=n_way, examples=100, split=split, normalize=True,
    #                                             data_len=DIM, SNR=None, label=True, set='sa')
    # _, _, test_x, test_y = generator.SA_37way(examples=200, split=0, way=n_way, data_len=DIM,
    #                                           normalize=True, label=True)

    # CW2SQ
    train_x, train_y, _, _ = generator.CW_cross(way=n_way, examples=50, split=split, normalize=True,
                                                data_len=DIM, SNR=None, label=True, set='sq')
    _, _, test_x, test_y = generator.SQ_37way(examples=100, split=0, way=n_way, data_len=DIM,
                                              normalize=True, label=True)
    # train_x, train_y, test_x, test_y = generator.EB_3_13way(examples=200, split=split,
    #                                                         way=n_way, data_len=DIM,
    #                                                         order=3, normalize=True, label=True)

    n_class = train_x.shape[0]
    assert n_class == n_way
    tar_x = test_x
    tar_y = test_y
    print('train_x Shape:', train_x.shape)
    print('test_x Shape:', test_x.shape)
    print('tar_x Shape:', tar_x.shape)
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    tar_x, tar_y = torch.from_numpy(tar_x).float(), torch.from_numpy(tar_y).long()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        # if os.path.exists(save_path):
        #     print('The training file exists：%s' % save_path)
        # else:
        train(net=net, save_path=save_path, train_x=train_x, train_y=train_y,
              tar_x=tar_x, tar_y=tar_y,
              ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            train_x = torch.cat((train_x, train_x), dim=1) if ob_domain else None
            # shot = 50 if ob_domain else shot
            save_path = r"G:\model_save\revised_DASMN\CW2SQ\CNN\CNN_cw2sq_30\epoch20"
            test(save_path=save_path, save_fig_path=fig_path, test_x=test_x, test_y=test_y, src_x=train_x,
                 n_way=n_way, shot=shot, eval=True)
            test(save_path=save_path, save_fig_path=fig_path, test_x=test_x, test_y=test_y, src_x=train_x,
                 n_way=n_way, shot=shot, eval=False)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':
    # 2020.11.4 DASMN paper revised.
    save_dir = r'G:\model_save\revised_DASMN\CW2SQ\CNN'
    path = os.path.join(save_dir, r'CNN_cw2sq_30')
    print('The model path\n', path)
    # path = os.path.join(save_dir, model_name)
    way = 3
    # n_shot = 5  # for time computation
    n_shot = 30  # 5 for training; 30 for t-SNE
    split = 30  # split the data, split=>train, the rest=>test

    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        os.mkdir(path)  # 如果要在训练过程中保存多个模型文件，请先建立文件夹
    fig_path = r"C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_tsne\CNN\CNN_Csq_30points.svg"
    main(save_path=path, fig_path=fig_path, n_way=way, shot=n_shot, split=split,
         ls_threshold=1e-4, ob_domain=True)

    exit()
    # 2020.7.29  SSMN revision
    save_dir = r'C:\Users\20996\Desktop\SSMN_revision\training_model\CNN'
    path = os.path.join(save_dir, r'CNN_C2S_10s')
    print('The model path', path)

    # path = os.path.join(save_dir, model_name)
    way = 3
    # n_shot = 5  # for time computation and training
    n_shot = 30  # 5 -10 for training; 30 for t-SNE
    split = 40  # split the data, split=>train, the rest=>test

    if not os.path.exists(save_dir):
        print('Root dir [{}] does not exist.'.format(save_dir))
        exit()
    else:
        print('File exist?', os.path.exists(path))

    # main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=False)
    main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=True)

    plt.show()
