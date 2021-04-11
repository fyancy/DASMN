"""
For the revised DASMN.
yancy F. 2020/11/1
"""

import torch
import numpy as np
import visdom
import time
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from Model.encoder_model import MetricNet
from my_utils.training_utils import sample_task_tr, sample_task_te
from my_utils.init_utils import weights_init0, weights_init1, weights_init2, set_seed
from my_utils.visualize_utils import Visualize_v2
from my_utils.plot_utils import tSNE_plot
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
Load = [3, 0]
CB_NUM = 8
WEIGHTS_INIT = weights_init2  # init2 ans 1 are better.

running_params = dict(train_epochs=150, test_epochs=3,
                      train_episodes=30, test_episodes=100,
                      train_split=100, test_split=0,  # generally same with train_split.
                      )
# ==========================


class ProtoLearner:
    def __init__(self, n_way, n_support, n_query):
        self.way = n_way
        self.ns = n_support
        self.nq = n_query
        self.visualization = Visualize_v2(vis)
        self.domain_criterion = nn.NLLLoss()

        self.model = MetricNet(self.way, self.ns, self.nq, vis=vis, cb_num=CB_NUM).to(device)


    def plot_adaptation(self, x_s, x_t):
        x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
        # print('CW2SQ labels used for t-sne!')
        # labels = ['NC-s', 'IF-s', 'OF-s', 'NC-t', 'IF-t', 'OF-t']  # CW2SQ
        # print('CW2SA labels used for t-sne!')
        # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA
        tSNE_plot(x.cpu().detach().numpy(), shot=self.ns,
                  name=None, labels=None, n_dim=2)

    def model_training(self, src_tasks, tgt_tasks, model_path):
        self.model.train()
        self.model.apply(WEIGHTS_INIT)  # not recommended weights initialization

        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)  # lr初始值设为0.1-0.2
        c_optimizer = optimizer1  # for encoder
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.95)  # lr=lr∗gamma^epoch

        print('source set for training:', src_tasks.shape)
        print('target set for validation', tgt_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['train_epochs']
        episodes = running_params['train_episodes']
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
            for epi in range(episodes):
                support, query = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)
                tgt_v_s, tgt_v_q = sample_task_tr(tgt_tasks, self.way, self.ns, length=DIM)

                src_loss, src_acc, _ = self.model.forward(xs=support, xq=query, sne_state=False)
                draw = False
                loss = src_loss

                c_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                # nn.utils.clip_grad_norm_(parameters=self.d_classifier.parameters(), max_norm=0.5)
                # To clip the grads of d_classifier is Not recommended.
                c_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    tgt_loss, tgt_acc, _ = self.model.forward(xs=tgt_v_s, xq=tgt_v_q, sne_state=False)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls

                if (epi + 1) % 5 == 0:
                    self.visualization.plot([src_ls, tgt_ls], ['Source_cls', 'Target_cls'],
                                            counter=counter, scenario="proto_Cls Loss")
                    self.visualization.plot([src_ac, tgt_ac], ['C_src', 'C_tgt'],
                                            counter=counter, scenario="proto_Cls_Acc")
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

            # if isinstance(c_optimizer, torch.optim.SGD):
            c_scheduler.step()

            if ep + 1 >= CHECK_EPOCH and (ep + 1) % delta == 0:
                flag = input("Shall we stop the training? Y/N\n")
                if flag == 'y' or flag == 'Y':
                    print('Training stops!(manually)')
                    new_path = os.path.join(model_path, f"final_epoch{ep+1}")
                    self.save(new_path, running_params['train_epochs'])
                    break
                else:
                    flag = input(f"Save model at epoch {ep+1}? Y/N\n")
                    if flag == 'y' or flag == 'Y':
                        child_path = os.path.join(model_path, f"epoch{ep+1}")
                        self.save(child_path, ep+1)

        print("The total time: {:.5f} s\n".format(np.sum(times)))

    def test(self, tar_tasks, src_tasks=None, mask=False, model_eval=True):
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
        print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(epochs, episodes,
                                                                           episodes * epochs))
        counter = 0
        avg_acc_all = 0.
        avg_loss_all = 0.

        print('Model.eval() is:', not self.model.training)
        for ep in range(epochs):
            avg_acc_ep = 0.
            avg_loss_ep = 0.
            sne_state = False
            for epi in range(episodes):
                tar_s, tar_q = sample_task_te(tar_tasks, self.way, self.ns, length=DIM)
                if src_tasks is not None and self.ns > 1:
                    src_s, src_q = sample_task_te(src_tasks, self.way, self.ns, length=DIM)

                # sne_state = True if epi + 1 == episodes else False
                with torch.no_grad():
                    tar_loss, tar_acc, zq_t = self.model.forward(xs=tar_s, xq=tar_q, sne_state=sne_state)
                    if src_tasks is not None and self.ns > 1:
                        _, _, zq_s = self.model.forward(xs=src_s, xq=src_s, sne_state=False)
                        if mask:
                            _, _, zq_t = self.model.forward(xs=src_q, xq=src_q, sne_state=False)

                tar_ls, tar_ac = tar_loss.cpu().item(), tar_acc.cpu().item()
                avg_acc_ep += tar_ac
                avg_loss_ep += tar_ls

                self.visualization.plot([tar_ac, tar_ls], ['Acc', 'Loss'],
                                        counter=counter, scenario="proto-Test")
                counter += 1

                # if (epi + 1) == episodes:
                #     print('[{}/{}, {}/{}] loss: {:.8f}\t acc: {:.8f}'.format(
                #         # ep + 1, epochs, epi + 1, episodes, tar_ls, tar_ac))
                #     if src_tasks is not None and self.ns > 1:
                #         self.plot_adaptation(zq_s, zq_t)

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
                 }
        torch.save(state, filename)
        print('This model is saved at [%s]' % filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state'])
        print('Load Encoder successfully from [%s]' % filename)


def train_operate(way, ns, nq, save_path, final_test=True, load_path=None):
    set_seed(0)
    nets = ProtoLearner(n_way=way, n_support=ns, n_query=nq)
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
                                data_len=DIM, normalize=True)  # 100/file, 2 files.

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
    print('Train ProtoNet!')
    nets.model_training(src, tar, save_path)

    # training 2: Turn off the GRL !!!!!!!
    # print('Train proto_2loss')
    # nets.train_proto_2loss(src, tar)

    if final_test:
        print('We test the model!')
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=True)
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=False)


def test_operate(way, ns, nq, path, eval_stats, ob_domain=False, num_domain=30):
    set_seed(0)
    if ob_domain:
        ns = nq = num_domain
        running_params['train_split'] = num_domain * 4

    model = ProtoLearner(n_way=way, n_support=ns, n_query=nq)

    # CW: NC, IF, OF, RoF
    src_tasks, _ = generator.CW_10way(way=way, order=Load[0], examples=200, split=running_params['train_split'],
                                      normalize=True, data_len=DIM, label=False)
    _, test_tasks = generator.CW_10way(way=way, order=Load[1], examples=200, split=0, normalize=True,
                                       data_len=DIM, label=False)

    # CW2SQ: NC, IF, OF
    # src_tasks, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'],
    #                                   normalize=True, data_len=DIM, SNR=None, label=False, set='sq')
    # _, test_tasks = generator.SQ_37way(examples=100, split=0, way=way, normalize=True,
    #                                    label=False, data_len=DIM)
    # _, test_tasks = generator.EB_3_13way(examples=running_params['test_samples'],
    #                                      split=running_params['test_split'],
    #                                      way=way, order=3, normalize=True, label=False, data_len=DIM)
    # CW2SA: NC, OF, RoF(Ball)
    # src_tasks, _ = generator.CW_cross(way=way, examples=100, split=running_params['train_split'], data_len=DIM,
    #                                   normalize=True, label=False, tgt_set='sq')
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
        model.test(test_tasks, src_tasks, model_eval=True)

    elif eval_stats == 'both':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=True)
        print('\n================Reloading the file====================')
        model.load(path)
        # Attention! Model.train() would change the trained weights(it's an invalid operation),
        # so we have to reload the trained file again.
        model.test(test_tasks, src_tasks, model_eval=False)

    elif eval_stats == 'no':
        model.load(path)
        model.test(test_tasks, src_tasks, model_eval=False)


if __name__ == "__main__":
    import os

    n_cls = 3  # 10, 3
    ns = nq = 5
    # m_f_root = r"G:\model_save\revised_DASMN\CW10\ProtoNets"
    # m_f_root = r"G:\model_save\revised_DASMN\CW2SA\ProtoNets"
    m_f_root = r"G:\model_save\revised_DASMN\CW2SQ\ProtoNets"

    # train:
    flag = input('Train? y/n\n')
    if flag == 'y' or flag == 'Y':
        save_path = os.path.join(m_f_root, 'proto8_cw2sq_' + str(running_params['train_split']))
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
        load_path = r"G:\model_save\revised_DASMN\CW10\ProtoNets\proto8_cw3to2_50\epoch10"
        test_operate(way=n_cls, ns=ns, nq=nq, path=load_path, eval_stats='both')
