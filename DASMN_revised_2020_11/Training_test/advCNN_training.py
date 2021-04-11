"""
CNN with adversarial domain adaptation.
yancy F. 2020/11/2
"""

import torch
import numpy as np
import visdom
import time
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from Model.CNN_model import CNN
from Model.advDA_net import DomainClassifier
# from my_utils.training_utils import sample_task_tr, sample_task_te
from my_utils.init_utils import weights_init0, weights_init1, weights_init2, set_seed
from my_utils.visualize_utils import Visualize_v2
# from my_utils.plot_utils import tSNE_plot
from Data_generate.DataLoadFn import DataGenFn

device = torch.device('cuda:0')
vis = visdom.Visdom(env='yancy_env')
generator = DataGenFn()

# ====== hyper params =======
CHN = 1
DIM = 1024  # 2048
CHECK_EPOCH = 10
WEIGHT_DECAY = 0  # 1e-5
CHECK_D = False  # check domain adaptation by t-SNE

WEIGHTS_INIT = weights_init2  # best for DASMN
# WEIGHTS_INIT = weights_init1
# WEIGHTS_INIT = weights_init0

# training params
Load = [3, 0]
adda_params = dict(alpha=1, gamma=10)
running_params = dict(train_epochs=200, test_epochs=3,
                      train_episodes=30, test_episodes=100,
                      train_split=100, test_split=0,  # generally same with train_split.
                      )


# ==========================


class CNNLearner:
    def __init__(self, n_way, n_support):
        self.way = n_way
        self.bsize = n_support  # batch size
        self.visualization = Visualize_v2(vis)
        self.domain_criterion = nn.NLLLoss()

        self.model = CNN(self.way, DIM, cb_num=8, drop_rate=0.3).to(device)
        self.d_classifier = DomainClassifier(DIM=DIM).to(device)

    @staticmethod
    def shuffle_dataset(a):
        # a: (nc, n, ...), only shuffle n samples in each class.
        new_a = torch.empty_like(a)
        index = torch.arange(a.shape[1]).unsqueeze(0).repeat(a.shape[0], 1).numpy()
        for k in index:
            np.random.shuffle(k)
        # print(index)
        for i in range(a.shape[0]):
            new_a[i] = a[i][index[i]]  # 对每个类单独 shuffle
        return new_a

    @staticmethod
    def get_constant(episodes, ep, epi):
        # total_steps = epochs * episodes
        total_steps = 3000  # 1000
        start_steps = ep * episodes
        p = float(epi + start_steps) / total_steps
        constant = torch.tensor(2. / (1. + np.exp(-adda_params['gamma'] * p)) - 1).to(device)
        return constant

    def domain_loss(self, src_x, tar_x, constant, draw=False):
        s_feature = self.model.get_features(src_x.reshape([-1, CHN, DIM]))
        t_feature = self.model.get_features(tar_x.reshape([-1, CHN, DIM]))
        src_dy, s_d_feat = self.d_classifier(s_feature, constant)
        tar_dy, t_d_feat = self.d_classifier(t_feature, constant)

        s_domain_y = torch.zeros(src_dy.shape[0]).long().to(device)
        t_domain_y = torch.ones(tar_dy.shape[0]).long().to(device)
        s_domain_loss = self.domain_criterion(src_dy, s_domain_y)
        t_domain_loss = self.domain_criterion(tar_dy, t_domain_y)
        s_domain_acc = torch.eq(src_dy.max(-1)[1], s_domain_y).float().mean()
        t_domain_acc = torch.eq(tar_dy.max(-1)[1], t_domain_y).float().mean()

        if draw:
            self.plot_adaptation(s_d_feat, t_d_feat)  # feature
            self.plot_adaptation(src_dy, tar_dy)  # label
            plt.show()
        return (s_domain_loss, t_domain_loss), (s_domain_acc, t_domain_acc)


    def train(self, src_tasks, src_labels, tgt_tasks, tgt_labels, model_path):
        self.model.apply(WEIGHTS_INIT), self.d_classifier.apply(WEIGHTS_INIT)
        self.model.train(), self.d_classifier.train()
        # for name, param in self.model.named_parameters():
        #     print(f"name: {name}, param: {param}")

        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=0.2, momentum=0.9)  # lr初始值设为0.1-0.2
        c_optimizer = optimizer1  # for encoder
        d_optimizer = torch.optim.RMSprop(self.d_classifier.parameters(), lr=1e-3, alpha=0.99)  # 跨域更好
        # d_optimizer = torch.optim.Adam(self.d_classifier.parameters(), lr=1e-3, weight_decay=1e-5)
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch

        tgt_tr_x = tgt_tasks[:, :src_tasks.shape[1]]  # N_src = N_tgt
        # tgt_tr_y = tgt_labels[:, :src_labels.shape[1]]
        print('source set for training:', src_tasks.shape)
        print('source labels for training:', src_labels.shape)
        print('target set for training', tgt_tr_x.shape)
        print('target set for validation', tgt_tasks.shape)
        print('source labels for validation:', tgt_labels.shape)
        print('batch size ==> ', self.bsize)

        epochs = running_params['train_epochs']
        episodes = src_tasks.shape[1] // self.bsize
        counter = 0
        draw = False
        avg_ls = torch.zeros([episodes])
        times = np.zeros([epochs])

        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        for ep in range(epochs):
            # if (ep + 1) <= 3 and CHECK_D:
            #     draw = True
            # elif 25 <= (ep + 1) <= 40 and CHECK_D:
            #     draw = True

            delta = 10 if (ep + 1) <= 30 else 5
            t0 = time.time()
            new_src_x = self.shuffle_dataset(src_tasks)
            new_tgt_x = self.shuffle_dataset(tgt_tr_x)
            new_tgt_v = self.shuffle_dataset(tgt_tasks)
            count = 0

            new_src_x, new_tgt_x, new_tgt_v = new_src_x.to(device), new_tgt_x.to(device), new_tgt_v.to(device)
            src_labels, tgt_labels = src_labels.to(device), tgt_labels.to(device)
            for epi in range(episodes):
                src_x, src_y = new_src_x[:, count:count + self.bsize], src_labels[:, count:count + self.bsize]
                tgt_x = new_tgt_x[:, count:count + self.bsize]  # tgt for training
                tgt_v_x, tgt_v_y = new_tgt_v[:, count:count + self.bsize], tgt_labels[:, count:count + self.bsize]
                count += self.bsize

                # training
                src_acc, src_loss = self.model.forward(src_x, src_y)
                constant = self.get_constant(episodes, ep, epi)  # constant: 0 => 1
                domain_loss, domain_acc = self.domain_loss(src_x, tgt_x, constant, draw=draw)
                draw = False

                d_loss = domain_loss[0] + domain_loss[1]
                loss = src_loss + adda_params['alpha'] * d_loss
                # loss = src_loss  # 验证是否对抗训练有用，证实确实有用！

                c_optimizer.zero_grad()
                d_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                # nn.utils.clip_grad_norm_(parameters=self.d_classifier.parameters(), max_norm=0.5)
                # To clip the grads of d_classifier is Not recommended.
                c_optimizer.step()
                d_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    tgt_acc, tgt_loss = self.model.forward(tgt_v_x, tgt_v_y)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls

                if (epi + 1) % 1 == 0:
                    self.visualization.plot([src_ls, tgt_ls], ['Source_cls', 'Target_cls'],
                                            counter=counter, scenario="advCNN_Cls Loss")
                    self.visualization.plot([src_ac, tgt_ac], ['C_src', 'C_tgt'],
                                            counter=counter, scenario="advCNN_Cls_Acc")
                    self.visualization.plot([domain_acc[0].cpu().item(), domain_acc[1].cpu().item()],
                                            label=['D_src', 'D_tgt'],
                                            counter=counter, scenario="advCNN_D_Acc")
                    self.visualization.plot([domain_loss[0].cpu().item(), domain_loss[1].cpu().item()],
                                            label=['Source_d', 'Target_d'],
                                            counter=counter, scenario="advCNN_D_Loss")
                    self.visualization.plot([loss.cpu().item()], label=['All_Loss'],
                                            counter=counter, scenario="advCNN_All_Loss")
                    counter += 1

                # if (epi + 1) % 10 == 0:
                #     print('[epoch {}/{}, episode {}/{}] => loss: {:.6f}, acc: {:.6f}'.format(
                #         ep + 1, epochs, epi + 1, episodes, src_ls, src_ac))
            # epoch
            t1 = time.time()
            times[ep] = t1 - t0
            print('[epoch {}/{}] time: {:.6f} Total: {:.6f}'.format(ep + 1, epochs, times[ep], np.sum(times)))
            ls_ = torch.mean(avg_ls).cpu()  # .item()
            print('[epoch {}/{}] avg_loss: {:.6f}\n'.format(ep + 1, epochs, ls_))
            if isinstance(c_optimizer, torch.optim.SGD):
                c_scheduler.step()  # ep // 5
            d_scheduler.step()  # ep // 5

            if ep + 1 >= CHECK_EPOCH and (ep + 1) % delta == 0:
                flag = input("Shall we stop the training? Y/N\n")
                if flag == 'y' or flag == 'Y':
                    print('Training stops!(manually)')
                    new_path = os.path.join(model_path, f"final_epoch{ep + 1}")
                    self.save(new_path, running_params['train_epochs'])
                    break
                else:
                    flag = input(f"Save model at epoch {ep + 1}? Y/N\n")
                    if flag == 'y' or flag == 'Y':
                        child_path = os.path.join(model_path, f"epoch{ep + 1}")
                        self.save(child_path, ep + 1)

            # self.visualization.plot(data=[1000 * optimizer.param_groups[0]['lr']],
            #                         label=['LR(*0.001)'], counter=ep,
            #                         scenario="SSMN_Dynamic params")
        print("The total time: {:.5f} s\n".format(np.sum(times)))

    def test(self, tgt_tasks, tgt_labels, src_tasks=None, src_labels=None, mask=False, model_eval=True):
        """
        :param mask:
        :param src_tasks: for t-sne
        :param tgt_tasks: target tasks [way, n, dim]
        :return:
        """
        if model_eval:
            self.model.eval()
        else:
            self.model.train()
        print('target set', tgt_tasks.shape)
        print('Batch size ==> ', self.bsize)

        epochs = running_params['test_epochs']
        episodes = tgt_tasks.shape[1] // self.bsize
        print(f'Start to test! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')

        counter = 0
        avg_acc_all = 0.
        avg_loss_all = 0.

        print('Model.eval() is:', not self.model.training)
        tgt_tasks, tgt_labels = tgt_tasks.to(device), tgt_labels.to(device)
        if src_tasks is not None:
            src_tasks, src_labels = src_tasks.to(device), src_labels.to(device)
        for ep in range(epochs):
            avg_acc_ep = 0.
            avg_loss_ep = 0.
            sne_state = False
            count = 0
            for epi in range(episodes):
                tgt_x, tgt_y = tgt_tasks[:, count:count + self.bsize], tgt_labels[:, count:count + self.bsize]
                if src_tasks is not None and self.bsize > 1:
                    src_x, src_y = src_tasks[:, count:count + self.bsize], src_labels[:, count:count + self.bsize]
                count += self.bsize


                # sne_state = True if epi + 1 == episodes else False
                with torch.no_grad():
                    tgt_acc, tgt_loss = self.model.forward(tgt_x, tgt_y)
                    # if src_tasks is not None and self.bsize > 1:
                    # _, _, zq_s = self.model.forward(xs=src_s, xq=src_s, sne_state=False)

                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_acc_ep += tgt_ac
                avg_loss_ep += tgt_ls

                self.visualization.plot([tgt_ac, tgt_ls], ['Acc', 'Loss'],
                                        counter=counter, scenario="advCNN Test")
                counter += 1

                # if (epi + 1) == episodes:
                #     print('[{}/{}, {}/{}] loss: {:.8f}\t acc: {:.8f}'.format(
                #         ep + 1, epochs, epi + 1, episodes, tgt_ls, tgt_ac))
                # if src_tasks is not None and self.ns > 1:
                #     self.plot_adaptation(zq_s, zq_t)

            # epoch
            avg_acc_ep /= episodes
            avg_loss_ep /= episodes
            avg_acc_all += avg_acc_ep
            avg_loss_all += avg_loss_ep
            print(f'[epoch {ep + 1}/{epochs}] avg_loss: {avg_loss_ep:.8f}\tavg_acc: {avg_acc_ep:.8f}')
        avg_acc_all /= epochs
        avg_loss_all /= epochs
        print('\n------------------------Average Result----------------------------')
        print('Average Test Loss: {:.6f}'.format(avg_loss_all))
        print('Average Test Accuracy: {:.6f}\n'.format(avg_acc_all))
        vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not self.model.training, avg_acc_all),
                 win='Eval:{} Test result'.format(not self.model.training))

    def save(self, filename, epoch):
        if os.path.exists(filename):
            filename += '(1)'
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'discriminator': self.d_classifier.state_dict(),
                 }
        torch.save(state, filename)
        print('This model is saved at [%s]' % filename)

    def load(self, filename, e=True, d=False):
        state = torch.load(filename)
        if e:
            self.model.load_state_dict(state['model_state'])
            print('Load Encoder successfully from [%s]' % filename)
        if d:
            self.d_classifier.load_state_dict(state['discriminator'])
            print('Load discriminator successfully from [%s]' % filename)


def train_operate(way, ns, save_path, final_test=True, load_path=None):
    set_seed(0)
    nets = CNNLearner(n_way=way, n_support=ns)
    if load_path is not None:  # 若加载路径不为空，则默认：模型微调
        nets.load(load_path)

    # CW: NC, IF, OF, RoF
    # src_x, src_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=100, split=running_params['train_split'],
    #                                         normalize=True, data_len=DIM, label=True)
    # _, _, tgt_x, tgt_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                         data_len=DIM, label=True)

    # CW2SQ: NC, IF, OF
    # src_x, src_y, _, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
    #                                         normalize=True, label=True, tgt_set='sq')
    # _, _, tgt_x, tgt_y = generator.SQ_37way(examples=100, split=0, way=way, label=True,
    #                                         data_len=DIM, normalize=True)

    # src, tar = generator.EB_3_13way(examples=running_params['train_samples'],
    #                                 split=running_params['train_split'],
    #                                 way=way, order=3, normalize=True, label=False)

    # CW2SA: NC, OF, RoF(Ball)
    src_x, src_y, _, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
                                            normalize=True, label=True, tgt_set='sa')
    _, _, tgt_x, tgt_y = generator.SA_37way(examples=200, split=0, way=way, label=True,
                                            data_len=DIM, normalize=True)

    # ---------------self testing on SA----------------------------------------------------
    # src, tar = generator.SA_37way(examples=200, split=running_params['train_split'],
    #                               way=way, normalize=True, label=False, overlap=True)

    # print('Train proto_1optimizer\n')  # 推荐
    # nets.train_proto_1op(src, tar)  # turn on the GRL

    # training 1:
    print('Train joint_training')  # 推荐 train with 2 optimizers
    nets.train(src_x, src_y, tgt_x, tgt_y, save_path)  # turn on the GRL

    # training 2: Turn off the GRL !!!!!!!
    # print('Train proto_2loss')
    # nets.train_proto_2loss(src, tar)

    if final_test:
        print('We test the model!')
        nets.test(tgt_tasks=tgt_x, tgt_labels=tgt_y, src_tasks=None, model_eval=True)
        nets.test(tgt_tasks=tgt_x, tgt_labels=tgt_y, src_tasks=None, model_eval=False)


def test_operate(way, ns, path, eval_stats, ob_domain=False, num_domain=30):
    set_seed(0)
    if ob_domain:
        ns = nq = num_domain
        running_params['train_split'] = num_domain * 4

    model = CNNLearner(n_way=way, n_support=ns)

    # CW: NC, IF, OF, RoF
    # src_x, src_y, _, _ = generator.CW_10way(way=way, order=Load[0], examples=100, split=running_params['train_split'],
    #                                         normalize=True, data_len=DIM, label=True)
    # _, _, tgt_x, tgt_y = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                         data_len=DIM, label=True)

    # CW2SQ: NC, IF, OF
    # src_x, src_y, _, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'],
    #                                         normalize=True, data_len=DIM, label=True, tgt_set='sq')
    # _, _, tgt_x, tgt_y = generator.SQ_37way(examples=100, split=0, way=way, normalize=True,
    #                                         label=True, data_len=DIM)
    # _, test_tasks = generator.EB_3_13way(examples=running_params['test_samples'],
    #                                      split=running_params['test_split'],
    #                                      way=way, order=3, normalize=True, label=False, data_len=DIM)
    # CW2SA: NC, OF, RoF(Ball)
    src_x, src_y, _, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
                                            normalize=True, label=True, tgt_set='sa')
    _, _, tgt_x, tgt_y = generator.SA_37way(examples=200, split=0, way=way, label=True,
                                            data_len=DIM, normalize=True)

    # src_tasks = src_tasks if ob_domain else None
    src_x = src_x if ob_domain else None
    src_y = src_y if ob_domain else None
    running_params['test_episodes'] = 10 if ob_domain else 100
    # if you do not want to observe the src and tgt results together
    print('test_task shape: ', tgt_x.shape)
    if ob_domain:
        print('src_task shape: ', src_x.shape)

    if eval_stats == 'yes':
        model.load(path)
        model.test(tgt_x, tgt_y, src_x, src_y, model_eval=True)

    elif eval_stats == 'both':
        model.load(path)
        model.test(tgt_x, tgt_y, src_x, src_y, model_eval=True)
        print('\n================Reloading the file====================')
        model.load(path)
        # Attention! Model.train() would change the trained weights(it's an invalid operation),
        # so we have to reload the trained file again.
        model.test(tgt_x, tgt_y, src_x, src_y, model_eval=False)

    elif eval_stats == 'no':
        model.load(path)
        model.test(tgt_x, tgt_y, src_x, src_y, model_eval=False)


if __name__ == "__main__":
    import os

    n_cls = 3  # 10, 3
    ns = nq = 5
    m_f_root = r"G:\model_save\revised_DASMN\CW2SA\advCNN"

    # train:
    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        save_path = os.path.join(m_f_root, 'advcnn_cw2sa' + str(running_params['train_split']))
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
        train_operate(way=n_cls, ns=ns, save_path=save_path, final_test=True)

    # test:
    flag = input('Test? y/n\n')
    if flag == 'y' or flag == 'Y':
        load_path = r"G:\model_save\revised_DASMN\CW2SQ\advCNN\advcnn_cw2sq100\epoch30"
        test_operate(way=n_cls, ns=ns, path=load_path, eval_stats='both')
