from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # step1:继承Exp_Basic,self.model = self._build_model().to(self.device)
        # step2:调用Exp_Long_Term_Forecast的_build_model()方法
        # step3:model = self.model_dict[self.args.model].Model(self.args).float()返回给self.model
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # 每一个xxformer.py文件中都有一个Model类用于创建model
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        vali_data:(34464,46080)构成的dataset
        vali_loader:(34464,46080)构成的dataloader
        criterion:mse
        """
        total_loss = []
        self.model.eval()
        # 验证阶段无需设置epoch,直接把所有验证数据进行一次计算loss即可
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # batch_x:(32,96,7)  [0,96]
                # batch_y:(32,144,7) [48,192],后续会取出前48个元素作为,即[48,96]作为decoder的输入
                # batch_x_mark:      [0,96]
                # batch_y_mark:      [48,192]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        # 验证结束后重新设置为训练模式
        self.model.train()
        # 返回一个epoch验证的平均loss
        return total_loss

    def train(self, setting):
        """
        输入(batch,96,7),经过模型得到(batch,192,7)的输出,然后取倒数96行作为[96,192]的预测结果
        """
        # 构建dataset和dataloader
        # (0,34560)
        train_data, train_loader = self._get_data(flag='train')
        # (34464,46080)
        vali_data, vali_loader = self._get_data(flag='val')
        # (45984,57600)
        test_data, test_loader = self._get_data(flag='test')

        # 构建模型保存路径,setting是详细信息构成的字符串
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # 构建early-stopping,优化器和损失(默认adam和mse)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 默认不使用混合精度训练
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 默认10个epochs
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # batch_x:(32,96,7)       [0,96]
                # batch_y:(32,144,7)      [48,192]
                # batch_x_mark:(32,96,4)  [0,96]
                # batch_y_mark:(32,144,4) [48,192]
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # (32,96,7)的全零
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()# dec_inp:(32,96,7)
                # (32,48,7)和(32,96,7)合并为(32,144,7)[batch_y的前48行数据+96行全零]
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    # 自动进行混合精度训练
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            # x_enc: (32, 96, 7)
                            # x_mark_enc: (32, 96, 4)
                            # x_dec: (32, 144, 7):0-48行是原始时序的48-96的数据,48-144为96个0
                            # x_mark_dec: (32, 144, 4):0-144行原始时序48-144行的周期特征
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # (32,96,7)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # 是否为多变量预测单变量,f_dim=0
                        f_dim = -1 if self.args.features == 'MS' else 0
                        # outputs:(32,96,7)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # batch_y:(32,96,7)
                        # 分析:对于样本1而言,输入模型的是[0,96],预测部分是[96,192]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 计算损失
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    # 默认不进行混合精度也不进行output_attention,直接运行到这里
                    else:
                        # batch_x:(32,96,7)
                        # batch_x_mark:(32,96,4)
                        # dec_inp:(32,144,7)
                        # batch_y_mark:(32,144,4)
                        # outputs:(32,96,7)
                        # 如果是Non-stationary,输出(32,144,7),但是后续会取出最后96行作为预测结果的
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # 是否为多变量预测单变量,f_dim=0
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs:(32,96,7)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y:(32,96,7)
                    # 分析:对于样本1而言,输入模型的是[0,96],预测部分是[96,192]
                    # Q:为什么batch_y要设置为[48,192]而不是[96,192]?是因为别的任务会有不同吗？
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # outputs和batch_y均为(32,96,7)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 每迭代100次,输出一次信息
                if (i + 1) % 100 == 0:
                    # 输出迭代次数,epoch和loss
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算100个iter的平均时间
                    speed = (time.time() - time_now) / iter_count
                    # 计算预计剩余训练时间=一次iter的平均速度*剩余的iter数量(train_steps表示一次epoch的iter次数)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # 每100次迭代,重新记时
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 反向传播更新梯度
                    loss.backward()
                    model_optim.step()

            # 每一个epoch结束后,输出当前epoch时间,并且进行测试和验证
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 截止到目前的所有iter的平均loss
            train_loss = np.average(train_loss)
            # 计算验证集的一个epoch的平均loss
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # 计算测试集的一个epoch的平均loss,调用接口是一致的,只需要传入不同的数据即可
            test_loss = self.vali(test_data, test_loader, criterion)

            # 输出每个epoch下的训练,验证,测试损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 记录是否需要early-stopping,并且将最低loss的模型保存到checkpoint
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 所有epoch训练完成后,将验证集上loss最低的模型参数加载出来,赋给model,最后train函数返回self.model
        # 注意,下面的best_model_path是在early_stopping中被保存的
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 每次调用完self.train后,会返回验证集上的最佳模型作为self.model,因此纵使不执行if test,下面的model也是最佳model
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
