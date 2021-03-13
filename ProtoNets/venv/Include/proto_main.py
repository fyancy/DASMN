import torch
import os
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import visdom
import time
from proto_data_utils.my_utils import Euclidean_Distance, t_sne, umap_fun2, plot_confusion_matrix
from models.proto_model import Protonet
from proto_data_utils.Data_generator_normalize import data_generate
from proto_data_utils.train_utils import weights_init, weights_init2
from proto_data_utils.train_utils import set_seed, sample_task, sample_task_te
from proto_data_utils.plot_utils import tSNE_fun

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vis = visdom.Visdom(env='yancy_env')

initialization = weights_init2
generator = data_generate()
CHN = 1
# DIM = 2048  # 2048
DIM = 1024  # 1024 in DASMN paper
Tr_EPOCHS = 100
Te_EPOCHS = 2
CHECK_EPOCH = 10
Load = [3, 2]


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


def plot_adaptation(x_s, x_t):
    x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
    print('CW2SQ labels used for t-sne!')
    labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
    tSNE_fun(x.cpu().detach().numpy(), shot=ns, name=None, labels=labels, n_dim=2)


def train(net, save_path, train_x, tar_x, ls_threshold=1e-5,
          n_way=3, n_episodes=30, shot=3, skip_lr=0.005):
    net.train()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # lr初始值设为0.1
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # lr=lr∗gamma^epoch

    n_shot = n_query = shot
    n_examples = train_x.shape[1]
    n_class = train_x.shape[0]
    assert n_class >= n_way
    assert n_examples >= n_shot + n_query

    print('train_data set Shape:', train_x.shape)
    print('n_way=>', n_way, 'n_shot=>', n_shot, ' n_query=>', n_query)
    print("---------------------Training----------------------\n")
    counter = 0
    opt_flag = False
    avg_ls = torch.zeros([n_episodes]).to(device)
    n_epochs = Tr_EPOCHS
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            support, query = sample_task(train_x, n_way, n_shot, DIM=DIM)
            t_s, t_q = sample_task(tar_x, n_way, n_shot, DIM=DIM)
            losses, ls_ac, _, _, _ = net.forward(support, query)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            with torch.no_grad():
                net.eval()
                _, t_ls_ac, _, _, _ = net.forward(t_s, t_q)
                net.train()

            ls, ac = ls_ac['loss'], ls_ac['acc']
            t_ls, t_ac = t_ls_ac['loss'], t_ls_ac['acc']
            avg_ls[epi] = ls
            if (epi + 1) % 5 == 0:
                vis.line(Y=[[ls, t_ls]], X=[counter],
                         update=None if counter == 0 else 'append', win='proto_Loss',
                         opts=dict(legend=['src_loss', 'tar_loss'], title='proto_Loss'))
                vis.line(Y=[[ac, t_ac]], X=[counter],
                         update=None if counter == 0 else 'append', win='proto_Acc',
                         opts=dict(legend=['src_acc', 'tar_acc'], title='proto_Acc'))
                counter += 1
            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}' \
                      .format(ep + 1, n_epochs, epi + 1, n_episodes, ls, ac))

        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))
        scheduler.step()

        # if ep + 1 >= CHECK_EPOCH and (ep + 1) % 5 == 0:
        #     order = input("Shall we stop training now? (Epoch {}) Y/N\n".format(ep + 1))
        #     order = order == 'Y' or order == 'y'
        # else:
        #     order = False

        # if ls_ < skip_lr and opt_flag is False:
        #     optimizer = optimizer2
        #     print('============Optimizer Switch==========')
        #     opt_flag = True
        # if (ls_ <= ls_threshold and ep + 1 >= 50) and order:
        #     print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
        #     break
        # elif ls_ < 0.5 * ls_threshold and order:
        #     print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
        #     break
        # elif order:
        #     print('Stop manually!')
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
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, save_fig_path, test_x, src_x=None, scenario='test_proto', m_spd=False,
         n_way=3, shot=2, eval_=False, n_episodes=100):
    # --------------------修改-------------------------
    net = Protonet().to(device)
    net.load_state_dict(torch.load(save_path))
    print("Load the model Successfully！\n%s" % save_path)
    net = net.eval() if eval_ else net.train()
    print('Model.eval() is:', not net.training)
    n_s = n_q = shot

    # n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert n_class >= n_way

    print('tgt_data set Shape:', test_x.shape)
    if src_x is not None:
        print('src_data set Shape:', src_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, DIM))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, DIM))
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    counter = 0
    avg_time = []
    n_epochs = Te_EPOCHS
    print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(n_epochs, n_episodes,
                                                                       n_episodes * n_epochs))

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        time_ep = []
        sne_state = False
        for epi in range(n_episodes):
            # [Nc, n_spd, num_each_way, 2048] for multi-speed
            if m_spd:
                support = []
                query = []
                for i in range(test_x.shape[1]):
                    # print(test_x[:, i].shape)
                    s, q = sample_task_te(test_x[:, i], n_way, shot, DIM=DIM)
                    support.append(s)
                    query.append(q)
                support = torch.cat(support, dim=1)
                query = torch.cat(query, dim=1)
            else:
                support, query = sample_task_te(test_x, n_way, shot, DIM=DIM)

            if src_x is not None and shot > 1:
                src_s, src_q = sample_task_te(src_x, n_way, shot, DIM=DIM)

            if ep + epi == 0:
                print('Support shape ', support.shape)

            # sne_state = True if epi == n_episodes - 1 else False
            t0 = time.time()
            with torch.no_grad():
                _, ls_ac, zq_t, yt, yp = net.forward(xs=support, xq=query, vis=vis, sne_state=sne_state)
                if src_x is not None and shot > 1:
                    _, _, zq_s, yt, yp = net.forward(xs=src_s, xq=src_q, vis=vis, sne_state=False)
            t1 = time.time()

            ls, ac = ls_ac['loss'], ls_ac['acc']
            avg_acc += ac
            avg_loss += ls
            time_ep.append(t1 - t0)
            vis.line(Y=[[ac, ls]], X=[counter],
                     update=None if counter == 0 else 'append', win=scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=scenario))
            counter += 1

            # if (epi + 1) % 10 == 0 and shot == 10:
            #     if src_x is not None:
            #         plot_adaptation(zq_s, zq_t)
            #         plt.show()
            #         order = input('Save fig? Y/N\n')
            #         if order == 'y' or order == 'Y':
            #             plot_adaptation(zq_s, zq_t)
            #             new_path = check_creat_new(save_fig_path)
            #             plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
            #             print('Save t-SNE.eps to \n', new_path)
            #     else:
            #         y = torch.arange(0, n_class).reshape(n_class, 1).repeat(1, n_q).long().reshape(-1)
            #         t_sne(input_data=zq_t.cpu().detach().numpy(),
            #               input_label=y.cpu().detach().numpy(), classes=n_way, path=save_path)
            # plot_confusion_matrix(y_true=yt.cpu().numpy(),
            #                       y_pred=yp.cpu().numpy(), path=save_path)
            # if (epi + 1) == episodes:
            plot_adaptation(zq_s, zq_t)
            plt.show()
            order = input('Save fig? Y/N\n')
            if order == 'y' or order == 'Y':
                plot_adaptation(zq_s, zq_t)
                new_path = check_creat_new(save_fig_path)
                plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
                print('Save t-SNE.eps to \n', new_path)

        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        avg_time.append(np.mean(time_ep) / shot * 200)
        print('[{}/{}]\tavg_time: {:.4f} s\tavg_loss: {:.6f}\tavg_acc: {:.4f}'.
              format(ep + 1, n_epochs, avg_time[-1], avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not net.training, avg_acc_),
             win='Eval:{} Test result'.format(not net.training))
    print('------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.4f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}'.format(avg_loss_))
    print('Average Test Time: {:.4f} s\n'.format(np.mean(avg_time)))


def main(save_path, n_way=3, shot=2, split=20, ls_threshold=1e-5, ob_domain=False):
    set_seed(0)
    net = Protonet().to(device)
    net.apply(initialization)  # 提升性能：对proto推荐使用手动初始化weights_init2
    print('%d GPU is available.' % torch.cuda.device_count())
    n_s = n_q = shot
    # if ob_domain:
    #     n_s = n_q = 50
    #     split = 50

    # CW: NC, IF, OF, RoF
    # m_spd = False
    # train_x, test_x = generator.Cs_4way(way=n_way, examples=50, split=split,
    #                                     normalize=True, data_len=DIM,
    #                                     label=False, shuffle=True)
    # SQ7
    # train_x, test_x = generator.SQ_37way(way=n_way, examples=100, split=split, shuffle=False,
    #                                      data_len=DIM, normalize=True, label=False)

    # m_spd = False
    # train_x, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False)
    # _, test_x = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                data_len=DIM, SNR=None, label=False)

    # CW2SQ: NC, IF, OF
    # m_spd = False
    # train_x, _ = generator.CW_cross(way=n_way, examples=50, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False, set='sq')
    # _, test_x = generator.SQ_37way(examples=100, split=0, data_len=DIM,
    #                                way=way, normalize=True, label=False)

    # speed
    # m_spd = True
    # _, test_x = generator.SQ_spd(examples=100, split=0, way=way, normalize=True, label=False)

    # CW2SA: NC, OF, RoF(Ball)
    # m_spd = False
    # train_x, _ = generator.CW_cross(way=n_way, examples=100, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False, set='sa')
    # _, test_x = generator.SA_37way(examples=200, split=0, way=way, data_len=DIM,
    #                                normalize=True, label=False)
    # speed
    # m_spd = True
    # _, test_x = generator.SA_spd(examples=200, split=0, way=n_way,
    #                              normalize=False, overlap=True, label=False)

    # EB data
    train_x, test_x = generator.EB_3_13way(examples=200, split=split, way=way, data_len=DIM,
                                           order=3, normalize=True, label=False)

    n_class = train_x.shape[0]
    assert n_class == n_way
    # tar_x = test_x[:, :train_x.shape[1]]
    tar_x = test_x

    print('train_data shape:', train_x.shape)
    print('test_data shape:', test_x.shape)
    print('target_data shape:', tar_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, DIM))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, DIM))

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        print('hhhhhh')
        # if os.path.exists(save_path):
        #     print('The training file exists：%s' % save_path)
        # else:
        train(net=net, save_path=save_path, train_x=train_x, tar_x=tar_x,
              ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            # if u want to observe the results of domain adaptation
            train_x = np.concatenate((train_x, train_x), axis=1) if ob_domain else None
            # shot = 50 if ob_domain else shot
            test(save_path=save_path, test_x=test_x, src_x=train_x, n_way=n_way,
                 shot=shot, eval_=True, n_episodes=100, m_spd=m_spd)
            exit()
            test(save_path=save_path, test_x=test_x, src_x=train_x, n_way=n_way,
                 shot=shot, eval_=False, n_episodes=100, m_spd=m_spd)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


def main_paper_dasmn(save_path, fig_path, n_way=3, shot=2, split=20, ls_threshold=1e-5, ob_domain=False):
    set_seed(0)
    net = Protonet().to(device)
    net.apply(initialization)  # 提升性能：对proto推荐使用手动初始化weights_init2
    print('%d GPU is available.' % torch.cuda.device_count())
    n_s = n_q = shot
    # if ob_domain:
    #     n_s = n_q = 50
    #     split = 50

    # CW: NC, IF, OF, RoF
    # m_spd = False
    # train_x, test_x = generator.Cs_4way(way=n_way, examples=50, split=split,
    #                                     normalize=True, data_len=DIM,
    #                                     label=False, shuffle=True)
    # SQ7
    # train_x, test_x = generator.SQ_37way(way=n_way, examples=100, split=split, shuffle=False,
    #                                      data_len=DIM, normalize=True, label=False)

    # m_spd = False
    # train_x, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False)
    # _, test_x = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                data_len=DIM, SNR=None, label=False)

    # CW2SQ: NC, IF, OF
    m_spd = False
    train_x, _ = generator.CW_cross(way=n_way, examples=50, split=split, normalize=True,
                                    data_len=DIM, SNR=None, label=False, set='sq')
    _, test_x = generator.SQ_37way(examples=100, split=0, data_len=DIM,
                                   way=n_way, normalize=True, label=False)

    # speed
    # m_spd = True
    # _, test_x = generator.SQ_spd(examples=100, split=0, way=way, normalize=True, label=False)

    # CW2SA: NC, OF, RoF(Ball)
    # m_spd = False
    # train_x, _ = generator.CW_cross(way=n_way, examples=100, split=split, normalize=True,
    #                                 data_len=DIM, SNR=None, label=False, set='sa')
    # _, test_x = generator.SA_37way(examples=200, split=0, way=n_way, data_len=DIM,
    #                                normalize=True, label=False)
    # train_x, test_x = generator.SA_37way(examples=200, split=50, way=n_way, data_len=DIM,
    #                                      normalize=True, label=False)  # 效果奇差；迁移可。
    # speed
    # m_spd = True
    # _, test_x = generator.SA_spd(examples=200, split=0, way=n_way,
    #                              normalize=False, overlap=True, label=False)

    # EB data
    # train_x, test_x = generator.EB_3_13way(examples=200, split=split, way=way, data_len=DIM,
    #                                        order=3, normalize=True, label=False)

    n_class = train_x.shape[0]
    assert n_class == n_way
    # tar_x = test_x[:, :train_x.shape[1]]
    tar_x = test_x

    print('train_data shape:', train_x.shape)
    print('test_data shape:', test_x.shape)
    print('target_data shape:', tar_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, DIM))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, DIM))

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        # if os.path.exists(save_path):
        #     print('The training file exists：%s' % save_path)
        # else:
        train(net=net, save_path=save_path, train_x=train_x, tar_x=tar_x,
              ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        save_path = r"G:\model_save\revised_DASMN\CW2SQ\ProtoNets\proto4_cw2sq_30\epoch15"
        if os.path.exists(save_path):
            # if u want to observe the results of domain adaptation
            train_x = np.concatenate((train_x, train_x), axis=1) if ob_domain else None
            # shot = 50 if ob_domain else shot
            # test(save_path=save_path, save_fig_path=fig_path, test_x=test_x, src_x=train_x, n_way=n_way,
            #      shot=shot, eval_=True, n_episodes=100, m_spd=m_spd)
            # exit()
            test(save_path=save_path, save_fig_path=fig_path, test_x=test_x, src_x=train_x, n_way=n_way,
                 shot=shot, eval_=False, n_episodes=100, m_spd=m_spd)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':  # train10, train13

    # 2020.11.4 DASMN paper revised.
    n_cls = 3  # 10, 3
    ns = nq = 5  # 5 for training; 30 for t-SNE
    m_f_root = r"G:\model_save\revised_DASMN\CW2SQ\ProtoNets"

    # train:
    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        save_path = os.path.join(m_f_root, 'proto4_cw2sq_30')
        fig_path = r"C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_tsne\ProtoNets4" \
                   r"\proto4_Csq.svg"

        if not os.path.exists(m_f_root):  # 首先判断路径存在与否
            print(f'File path does not exist:\n{m_f_root}')
            exit()
        if os.path.exists(save_path):
            print(f'File path [{save_path}] exist!')
            order = input(f'Go on? y/n\n')
            if order == 'y' or order == 'Y':
                save_path += '(1)'
                print(f"Go on with file\n{save_path}")
            else:
                exit()

        os.mkdir(save_path)  # 如果要在训练过程中保存多个模型文件，请先建立文件夹
        main_paper_dasmn(n_way=n_cls, shot=ns, save_path=save_path, fig_path=fig_path, split=30, ob_domain=True)

    # test:
    flag = input('Test? y/n\n')
    if flag == 'y' or flag == 'Y':
        load_path = r"G:\model_save\revised_DASMN\CW2SQ\DASMN\dasmn_cw2sq_10\final_epoch10"
        fig_path = r"C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_tsne\ProtoNets4" \
                   r"\proto4_Csq.svg"
        # test_operate(way=n_cls, ns=ns, nq=nq, path=load_path, eval_stats='no', ob_domain=True, fig_path=fig_path)
        main_paper_dasmn(n_way=n_cls, shot=ns, save_path=load_path, fig_path=fig_path, split=30, ob_domain=True)

    # 2020.7.29
    # save_dir = r'C:\Users\20996\Desktop\SSMN_revision\training_model\ProtoNets'
    # path = os.path.join(save_dir, r'ProtoNets_C2S_10s')
    # print('The model path\n', path)
    # way = 3
    # # n_shot = 5  # for time computation
    # n_shot = 10  # for t-SNE
    # split = 40
    # eval_ = ['yes', 'no', 'both']
    #
    # if not os.path.exists(save_dir):
    #     print('Root dir [{}] does not exist.'.format(save_dir))
    #     exit()
    # else:
    #     print('File exist?', os.path.exists(path))
    #
    # main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=True)
    # # main(save_path=path, n_way=way, shot=n_shot, split=split, ls_threshold=1e-4, ob_domain=False)
    # plt.show()
