"""
T-SNE 是神经网络分类任务过程或结果可视化的重要手段。
1. 一个训练得足够好的模型，此处进行测试，对特征进行可视化。 cw to sq, encoded features, src and tgt.
2. 体现出对抗训练过程的合理性，模型从0开始训练到epoch=30进行对比。cw to sq, FCL features, src and tgt.
冯勇，2020-11-4
"""

import torch
import numpy as np
import visdom
import time
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from Model.encoder_model import MetricNet
from Model.advDA_net import DomainClassifier
from my_utils.training_utils import sample_task_tr, sample_task_te
from my_utils.init_utils import weights_init0, weights_init1, weights_init2, set_seed
from my_utils.visualize_utils import Visualize_v2
from my_utils.plot_utils import tSNE_fun
from Data_generate.DataLoadFn import DataGenFn

device = torch.device('cuda:0')
vis = visdom.Visdom(env='yancy_env')
generator = DataGenFn()

# ====== hyper params =======
CHN = 1
DIM = 1024  # 2048
CHECK_EPOCH = 10
WEIGHT_DECAY = 0  # 1e-5
CHECK_D = True  # check domain adaptation by t-SNE
Load = [3, 0]
# WEIGHTS_INIT = weights_init2  # best for DASMN
# WEIGHTS_INIT = weights_init1
WEIGHTS_INIT = weights_init0  # better

adda_params = dict(alpha=1, gamma=10)
running_params = dict(train_epochs=150, test_epochs=3,
                      train_episodes=30, test_episodes=100,
                      train_split=30, test_split=0,  # generally same with train_split.
                      )
TSNE_FIG_PATH = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_tsne\DASMN\Discussion_DA'
TSNE_FIG_PATH_feat = TSNE_FIG_PATH + r'\feat.svg'
TSNE_FIG_PATH_label = TSNE_FIG_PATH + r'\label.svg'


def check_creat_new(path):
    if os.path.exists(path):
        split_f = os.path.split(path)
        new_f = os.path.join(split_f[0], split_f[1][:-4] + '(1).svg')
        new_f = check_creat_new(new_f)  # in case the new file exist
    else:
        new_f = path
    return new_f


# ==========================


class DASMNLearner:
    def __init__(self, n_way, n_support, n_query):
        self.way = n_way
        self.ns = n_support
        self.nq = n_query
        self.visualization = Visualize_v2(vis)
        self.domain_criterion = nn.NLLLoss()

        self.model = MetricNet(self.way, self.ns, self.nq, vis=vis, cb_num=8).to(device)
        self.d_classifier = DomainClassifier(DIM=DIM).to(device)

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
            # self.plot_adaptation(s_d_feat, t_d_feat)  # domain feature
            # self.plot_adaptation(src_dy, tar_dy)  # domain label
            # print('Domain Features:')
            # self.plot_adaptation_v2(s_d_feat, t_d_feat, TSNE_FIG_PATH_feat)  # domain feature
            print('Domain Labels:')
            self.plot_adaptation_v2(src_dy, tar_dy, TSNE_FIG_PATH_label)  # domain label
        return (s_domain_loss, t_domain_loss), (s_domain_acc, t_domain_acc)

    def plot_adaptation(self, x_s, x_t):
        x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
        # print('CW2SQ labels used for t-sne!')
        # labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
        # tSNE_fun(x.cpu().detach().numpy(), shot=self.ns, name=None, labels=labels, n_dim=2)

        print('CW2SA labels used for t-sne!')
        labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
        tSNE_fun(x.cpu().detach().numpy(), shot=self.ns, name=None, labels=labels, n_dim=2)

    def plot_adaptation_v2(self, x_s, x_t, save_fig_path):
        print('CW2SQ labels used for t-sne!')
        labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
        # print('CW2SA labels used for t-sne!')
        # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA

        x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
        tSNE_fun(x.cpu().detach().numpy(), shot=self.ns, name=None, labels=labels, n_dim=2)
        plt.show()
        order = input('Save fig? Y/N\n')
        if order == 'y' or order == 'Y':
            tSNE_fun(x.cpu().detach().numpy(), shot=self.ns, name=None, labels=labels, n_dim=2)
            new_path = check_creat_new(save_fig_path)
            plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
            print('Save t-SNE.eps to \n', new_path)

    def joint_training_2op(self, src_tasks, tgt_tasks, model_path):
        self.model.train(), self.d_classifier.train()
        self.model.apply(WEIGHTS_INIT), self.d_classifier.apply(WEIGHTS_INIT)

        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)  # lr初始值设为0.1-0.2
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # SGD is better
        c_optimizer = optimizer1  # for encoder
        d_optimizer = torch.optim.RMSprop(self.d_classifier.parameters(), lr=1e-3, alpha=0.99)  # 跨域更好
        # d_optimizer = torch.optim.Adam(self.d_classifier.parameters(), lr=1e-3, weight_decay=1e-5)
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)  # lr=lr∗gamma^epoch
        # =======
        # optional_lr = 0.01  # 经验参数：0.001~0.05: 0.02 [to SA/SQ]
        optional_lr = 0.01  # 经验参数: 0.1~0.2 [CW]
        # =======

        tar_tr = tgt_tasks[:, :src_tasks.shape[1]]
        print('source set for training:', src_tasks.shape)
        print('target set for training', tar_tr.shape)
        print('target set for validation', tgt_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['train_epochs']
        episodes = running_params['train_episodes']
        counter = 0
        draw = False  # t-SNE
        opt_flag = False
        avg_ls = torch.zeros([episodes])
        times = np.zeros([epochs])

        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        for ep in range(epochs):
            # if (ep + 1) <= 3 and CHECK_D:
            #     draw = True
            # elif 25 <= (ep + 1) <= 40 and CHECK_D:
            #     draw = True
            draw = True if 30 <= (ep + 1) <= 40 and CHECK_D else False

            delta = 10 if (ep + 1) <= 30 else 5
            t0 = time.time()
            for epi in range(episodes):
                support, query = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)
                tgt_s, _ = sample_task_tr(tar_tr, self.way, self.ns, length=DIM)
                tgt_v_s, tgt_v_q = sample_task_tr(tgt_tasks, self.way, self.ns, length=DIM)

                src_loss, src_acc, _ = self.model.forward(xs=support, xq=query, sne_state=False)
                constant = self.get_constant(episodes, ep, epi)
                domain_loss, domain_acc = self.domain_loss(support, tgt_s, constant, draw=draw)
                # draw = False  # t-SNE

                d_loss = domain_loss[0] + domain_loss[1]
                loss = src_loss + adda_params['alpha'] * d_loss

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
                    tgt_loss, tgt_acc, _ = self.model.forward(xs=tgt_v_s, xq=tgt_v_q, sne_state=False)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls

                if (epi + 1) % 5 == 0:
                    self.visualization.plot([src_ls, tgt_ls], ['Source_cls', 'Target_cls'],
                                            counter=counter, scenario="DASMN_Cls Loss")
                    self.visualization.plot([src_ac, tgt_ac], ['C_src', 'C_tgt'],
                                            counter=counter, scenario="DASMN_Cls_Acc")
                    self.visualization.plot([domain_acc[0].cpu().item(), domain_acc[1].cpu().item()],
                                            label=['D_src', 'D_tgt'],
                                            counter=counter, scenario="DASMN_D_Acc")
                    self.visualization.plot([domain_loss[0].cpu().item(), domain_loss[1].cpu().item()],
                                            label=['Source_d', 'Target_d'],
                                            counter=counter, scenario="DASMN_D_Loss")
                    self.visualization.plot([loss.cpu().item()], label=['All_Loss'],
                                            counter=counter, scenario="DASMN_All_Loss")
                    counter += 1

                # if (epi + 1) % 10 == 0:
                #     print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                #         ep + 1, epochs, epi + 1, episodes, src_ls, src_ac))
            # epoch
            t1 = time.time()
            times[ep] = t1 - t0
            print('[epoch {}/{}] time: {:.5f} Total: {:.5f}'.format(ep + 1, epochs, times[ep], np.sum(times)))
            ls_ = torch.mean(avg_ls).cpu()  # .item()
            print('[epoch {}/{}] avg_loss: {:.8f}\n'.format(ep + 1, epochs, ls_))
            if isinstance(c_optimizer, torch.optim.SGD):
                c_scheduler.step()  # ep // 5
            d_scheduler.step()  # ep // 5
            if ls_ < optional_lr and opt_flag is False:
                # if (ep + 1) >= 20 and opt_flag is False:
                c_optimizer = optimizer2
                #     # c_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                print('====== Optimizer Switch ======\n')
                opt_flag = True

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

    def test(self, tar_tasks, src_tasks, save_fig_path, mask=False, model_eval=True):
        """
        :param mask:
        :param src_tasks: for t-sne
        :param tar_tasks: target tasks [way, n, dim]
        :return:
        """
        if model_eval:
            self.model.eval()
        else:
            self.model.train()
        print('target set', tar_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['test_epochs']
        episodes = running_params['test_episodes']
        # episodes = tar_tasks.shape[1] // self.ns
        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        counter = 0
        avg_acc_all = 0.
        avg_loss_all = 0.

        print('Model.eval() is:', not self.model.training)
        for ep in range(epochs):
            avg_acc_ep = 0.
            avg_loss_ep = 0.
            sne_state = False
            for epi in range(episodes):
                # tar_s, tar_q = sample_task_te(tar_tasks, self.way, self.ns, length=DIM)
                # src_s, src_q = sample_task_te(src_tasks, self.way, self.ns, length=DIM)
                tar_s, tar_q = sample_task_tr(tar_tasks, self.way, self.ns, length=DIM)
                src_s, src_q = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)

                # sne_state = True if epi + 1 == episodes else False
                with torch.no_grad():
                    tar_loss, tar_acc, zq_t = self.model.forward(xs=tar_s, xq=tar_q, sne_state=sne_state)
                    _, _, zq_s = self.model.forward(xs=src_s, xq=src_s, sne_state=False)
                    if mask:
                        _, _, zq_t = self.model.forward(xs=src_q, xq=src_q, sne_state=False)

                tar_ls, tar_ac = tar_loss.cpu().item(), tar_acc.cpu().item()
                avg_acc_ep += tar_ac
                avg_loss_ep += tar_ls

                self.visualization.plot([tar_ac, tar_ls], ['Acc', 'Loss'],
                                        counter=counter, scenario="DASMN-Test")
                counter += 1

                # if (epi + 1) == episodes:
                print(f'[{ep + 1}/{epochs}, {epi + 1}/{episodes}]\ttar_loss: {tar_ls:.8f}\ttar_acc: {tar_ac:.8f}')
                self.plot_adaptation(zq_s, zq_t)
                plt.show()
                order = input('Save fig? Y/N\n')
                if order == 'y' or order == 'Y':
                    self.plot_adaptation(zq_s, zq_t)
                    new_path = check_creat_new(save_fig_path)
                    plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
                    print('Save t-SNE.eps to \n', new_path)

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


def train_operate(way, ns, nq, save_path, final_test=True, load_path=None):
    set_seed(0)
    nets = DASMNLearner(n_way=way, n_support=ns, n_query=nq)
    if load_path is not None:  # 若加载路径不为空，则默认：模型微调
        nets.load(load_path)

    # CW: NC, IF, OF, RoF
    # src, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=running_params['train_split'],
    #                             normalize=True, data_len=DIM, label=False)
    # _, tar = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                             data_len=DIM, label=False)

    # CW2SQ: NC, IF, OF
    src, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
                                normalize=True, label=False, tgt_set='sq')
    _, tar = generator.SQ_37way(examples=100, split=0, way=way, label=False,
                                data_len=DIM, normalize=True)

    # src, tar = generator.EB_3_13way(examples=running_params['train_samples'],
    #                                 split=running_params['train_split'],
    #                                 way=way, order=3, normalize=True, label=False)

    # CW2SA: NC, OF, RoF(Ball)
    # src, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'],
    #                             normalize=True, data_len=DIM, label=False, tgt_set='sa')
    # _, tar = generator.SA_37way(examples=200, split=0, way=way, data_len=DIM,
    #                             normalize=True, label=False)

    # ---------------self testing on SA----------------------------------------------------
    # src, tar = generator.SA_37way(examples=200, split=running_params['train_split'],
    #                               way=way, normalize=True, label=False, overlap=True)

    # print('Train proto_1optimizer\n')  # 推荐
    # nets.train_proto_1op(src, tar)  # turn on the GRL

    # training 1:
    print('Train joint_training')  # 推荐 train with 2 optimizers
    nets.joint_training_2op(src, tar, save_path)  # turn on the GRL

    # training 2: Turn off the GRL !!!!!!!
    # print('Train proto_2loss')
    # nets.train_proto_2loss(src, tar)

    if final_test:
        print('We test the model!')
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=True)
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=False)


def test_operate(way, ns, nq, path, eval_stats, fig_path, ob_domain=False, num_domain=30):
    set_seed(0)
    # if ob_domain:
    #     ns = nq = num_domain
    #     running_params['train_split'] = num_domain * 4

    model = DASMNLearner(n_way=way, n_support=ns, n_query=nq)

    # CW: NC, IF, OF, RoF
    # src_tasks, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=running_params['train_split'],
    #                                   normalize=True, data_len=DIM, label=False)
    # _, test_tasks = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
    #                                    data_len=DIM, label=False)

    # CW2SQ: NC, IF, OF
    src_tasks, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'],
                                      normalize=True, data_len=DIM, label=False, tgt_set='sq')
    _, test_tasks = generator.SQ_37way(examples=100, split=0, way=way, normalize=True,
                                       label=False, data_len=DIM)
    # _, test_tasks = generator.EB_3_13way(examples=running_params['test_samples'],
    #                                      split=running_params['test_split'],
    #                                      way=way, order=3, normalize=True, label=False, data_len=DIM)
    # CW2SA: NC, OF, RoF(Ball)
    # src_tasks, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
    #                                   normalize=True, label=False, tgt_set='sa')
    # _, test_tasks = generator.SA_37way(examples=200, split=0, way=way, normalize=True,
    #                                    label=False, data_len=DIM)

    # src_tasks = src_tasks if ob_domain else None
    src_tasks = src_tasks if ob_domain else None
    running_params['test_episodes'] = 10 if ob_domain else 100
    # if you do not want to observe the src and tgt results together
    print('test_task shape: ', test_tasks.shape)
    if ob_domain:
        print('src_task shape: ', src_tasks.shape)

    if eval_stats == 'yes':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=True, save_fig_path=fig_path)

    elif eval_stats == 'both':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=True, save_fig_path=fig_path)
        print('\n================Reloading the file====================')
        model.load(path)
        # Attention! Model.train() would change the trained weights(it's an invalid operation),
        # so we have to reload the trained file again.
        model.test(test_tasks, src_tasks, model_eval=False, save_fig_path=fig_path)

    elif eval_stats == 'no':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=False, save_fig_path=fig_path)


if __name__ == "__main__":
    import os

    n_cls = 3  # 10, 3
    ns = nq = 5  # 5 for training; 30 for t-SNE.
    m_f_root = r"G:\model_save\revised_DASMN\CW2SA\DASMN"

    # train:
    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        save_path = os.path.join(m_f_root, 'dasmn_cw2sq_' + str(running_params['train_split']))
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
        train_operate(way=n_cls, ns=ns, nq=nq, save_path=save_path, final_test=True)

    # test:
    flag = input('Test? y/n\n')
    if flag == 'y' or flag == 'Y':
        load_path = r"G:\model_save\revised_DASMN\CW2SA\DASMN\dasmn_cw2sa_100\final_epoch120"
        fig_path = r"C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_tsne\DASMN\DASMN_Csa30_2.svg"
        test_operate(way=n_cls, ns=ns, nq=nq, path=load_path, eval_stats='both', ob_domain=True, fig_path=fig_path)
