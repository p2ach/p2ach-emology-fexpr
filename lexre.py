# from datagen import TrainSet, ValidationSet, TestSet

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import time
from tqdm import tqdm
from model import FaceExprNet

from dataset import FaceExpr
from eval_peach.peach_compare import compare
from eval_peach.peach_compare import EMOTIONS
import pandas as pd

class ExpRecognition():
    def prepare_devices(self, gpu_ids, landmark_num=68,chns=3):
        str_ids = gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.landmark_num = landmark_num
        self.chns = chns
        # self.image_width = image_width

    def load_train_data(self, train_path, batch_size=128, num_workers=4):
        trainset = FaceExpr(src_dir=train_path)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers)
        # self.mean_landmark, valid_landmark_num = trainset.cal_mean_landmark()
        # print('mean landmark:')
        # print(self.mean_landmark)
        # print('valid landmark number:')
        # print(int(valid_landmark_num))
        # self.mean_landmark = torch.FloatTensor(self.mean_landmark).to(self.device)

        # training_data = FaceExpr(src_dir=args.train_path)
        # test_data = FaceExpr(src_dir=args.validation_path)
        # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    def load_validation_data(self, val_path, label_path, num_workers=4):
        validationset = FaceExpr(val_path)
        self.validation_loader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False,
                                                             num_workers=num_workers)

    # def load_test_data(self, image_path, num_workers=4):
    #     testset = TestSet(image_path)
    #     self.test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    def prepare_tool(self, start_lr=1e-2, learning_rate_decay_start=100, total_epoch=3000, model_path=None, \
                     beta=0.7, margin_1=0.5, margin_2=0.4, relabel_epoch=1800):
        # model

        # self.model = VGG('VGG19', landmark_num=self.landmark_num) # use VGG model
        self.model = FaceExprNet(landmark_num=self.landmark_num,chns=self.chns)  # use ResNet18 model

        if model_path is not None:
            assert (torch.cuda.is_available())
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model, self.gpu_ids)
            ck = torch.load(model_path)
            self.model.load_state_dict(ck['net'])
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model, self.gpu_ids)
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=start_lr)
        # loss function
        self.loss_fn = nn.MSELoss().to(self.device)

        # load related setting
        self.beta = beta
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.relabel_epoch = relabel_epoch

        # record messages
        self.start_lr = start_lr
        self.learning_rate_decay_start = max(0, learning_rate_decay_start)
        self.total_epoch = total_epoch

    def train(self, epoch):
        start = time.time()
        self.model.train()
        total_rr_loss = 0.0
        total_ce_loss = 0.0
        total_lm_loss = 0.0
        total_loss = 0.0
        total_num = 0
        class_total = list(0. for i in range(7))
        len_train_loader = len(self.train_loader)
        print("train_loader size", len_train_loader)

        for batch_idx, (img, label) in enumerate(tqdm(self.train_loader)):
            # if batch_idx>10:
            #     break
            img, label = \
                img.to(self.device).float(), label.to(self.device).float()
                # landmark.to(self.device).float(), have_landmark.to(self.device).long()

            # Self-attention Importance Weighting Module
            p_1, p_2, p_3, p_4, p_5, p_6, p_7 = self.model(img)

            '''SCN module in PyTorch.
            Reference:
            [1] Kai Wang, Xiaojiang Peng, Jianfei Yang, Shijian Lu, Yu Qiao
                Suppressing Uncertainties for Large-Scale Facial Expression Recognition. arXiv:2002.10392
            '''

            # Rank Regularization Module
            batch_size = img.shape[0]
            # tops = int(batch_size * self.beta)
            # _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            # _, down_idx = torch.topk(attention_weights.squeeze(), batch_size - tops, largest=False)
            # high_group = attention_weights[top_idx]
            # low_group = attention_weights[down_idx]
            # high_mean = torch.mean(high_group)
            # low_mean = torch.mean(low_group)
            # diff = low_mean - high_mean + self.margin_1
            #
            # # Rank Regularization Loss
            # if diff > 0.0:
            #     RR_loss = diff
            # else:
            #     RR_loss = 0.0

            # Cross Entropy Loss
            mse_loss_1 = self.loss_fn(p_1, torch.unsqueeze(label[:,0 ],1))
            mse_loss_2 = self.loss_fn(p_2, torch.unsqueeze(label[:,1 ],1))
            mse_loss_3 = self.loss_fn(p_3, torch.unsqueeze(label[:,2 ],1))
            mse_loss_4 = self.loss_fn(p_4, torch.unsqueeze(label[:,3 ],1))
            mse_loss_5 = self.loss_fn(p_5, torch.unsqueeze(label[:,4 ],1))
            mse_loss_6 = self.loss_fn(p_6, torch.unsqueeze(label[:,5 ],1))
            mse_loss_7 = self.loss_fn(p_7, torch.unsqueeze(label[:,6 ],1))

            loss = mse_loss_1+mse_loss_2+mse_loss_3+mse_loss_4+mse_loss_5+mse_loss_6+mse_loss_7
            # # Landmark Loss
            # land_2d += self.mean_landmark
            # LM_loss = torch.mean(torch.abs(land_2d - landmark) * have_landmark[:, :, None])
            #
            # # Whole Loss
            # # factor = 1.0 * (self.total_epoch - epoch) / self.total_epoch
            # loss = RR_loss + CE_loss + LM_loss
            for i in range(batch_size):
                #     # lbl = class_correct[i]
                #     if corrected == True:
                #         class_correct[label.detach().cpu().numpy().squeeze()] += 1
                #         all_class_correct += 1
                label_prob = label[i].detach().cpu().numpy().squeeze()
                label_prob = np.argmax(label_prob)

                class_total[int(label_prob)] += 1

            if epoch >= self.learning_rate_decay_start:
                lr = self.start_lr * (self.total_epoch - epoch) / (self.total_epoch - self.learning_rate_decay_start)
                self.set_lr(self.optimizer, lr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # total_rr_loss += RR_loss * batch_size
            total_ce_loss += loss.item() * batch_size
            # total_lm_loss += LM_loss.item() * batch_size
            # total_loss += loss.item() * batch_size
            total_num += batch_size

            # Relabeling Module
            # if epoch >= self.relabel_epoch:
                # sm_prob = torch.softmax(weighted_prob, dim=1)
                # prob_max, predicted_labels = torch.max(sm_prob, 1)
                # prob_gt = torch.gather(sm_prob, 1, label.view(-1, 1)).squeeze()
                # t_or_f = prob_max - prob_gt > self.margin_2
                # update_idx = t_or_f.nonzero().squeeze()
                # label_index = index[update_idx]
                # relabels = predicted_labels[update_idx]
                # self.train_loader.dataset.labels[label_index.cpu().numpy()] = relabels.cpu().numpy()
        end = time.time()

        print('epoch_' + str(epoch) + '\t loss: ' + '{:3.6f}, class_total {}'.format(total_ce_loss / total_num, class_total))

        #':\tspend ' + str(end - start) + 's'+ \

              # '\trr loss: ' + '{:3.6f}'.format(total_rr_loss / total_num) + \
              # '\tce loss: ' + '{:3.6f}'.format(total_ce_loss / total_num) + \
              # '\tlm loss: ' + '{:3.6f}'.format(total_lm_loss / total_num))

    def validation(self, epoch):
        start = time.time()
        self.model.eval()
        total_loss = 0.0
        total_num = 0
        # validation_path += (str(epoch) + '.txt')
        # file = open(validation_path, 'w')
        len_validation_size=len(self.validation_loader)
        all_class_correct = 0
        all_class_total = 0
        class_correct = list(0. for i in range(7))
        class_predict = list(0. for i in range(7))
        class_total = list(0. for i in range(7))
        class_accuracy = list(0. for i in range(7))

        print("len_validation_size", len_validation_size)

        validlines_label = []
        validlines_prediction = []
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(self.validation_loader):
                if batch_idx > 20000:
                    break
                img, label = \
                    img.to(self.device).float(), label.to(self.device).float()

                batch_size = img.shape[0]
                # print("batch_size", batch_size)

                # img, label = img.to(self.device).float(), label.to(self.device).long()
                # _, weighted_prob, _ = self.model(img)
                p_1, p_2, p_3, p_4, p_5, p_6, p_7 = self.model(img)



                # if len(label_list) !=7:
                #     print(label)
                #     pass
                # if len(list(predict.cpu().detach().numpy()[0])) !=7:
                #     print(predict)
                #     pass


                # if len(validlines_label)!=len(validlines_label):
                #     print("not match size of batch_idx", batch_idx)

                mse_loss_1 = self.loss_fn(p_1, torch.unsqueeze(label[:,0 ],1))
                mse_loss_2 = self.loss_fn(p_2, torch.unsqueeze(label[:,1 ],1))
                mse_loss_3 = self.loss_fn(p_3, torch.unsqueeze(label[:,2 ],1))
                mse_loss_4 = self.loss_fn(p_4, torch.unsqueeze(label[:,3 ],1))
                mse_loss_5 = self.loss_fn(p_5, torch.unsqueeze(label[:,4 ],1))
                mse_loss_6 = self.loss_fn(p_6, torch.unsqueeze(label[:,5 ],1))
                mse_loss_7 = self.loss_fn(p_7, torch.unsqueeze(label[:,6 ],1))

                total_loss += mse_loss_1.item()+mse_loss_2.item()+mse_loss_3.item()+mse_loss_4.item()+mse_loss_5.item()+mse_loss_6.item()+mse_loss_7.item() #* img.shape[0]
                total_num += img.shape[0]



                label_list = list(label.cpu().detach().numpy()[0])
                p_1 = list(p_1.cpu().detach().numpy()[0])
                p_2 = list(p_2.cpu().detach().numpy()[0])
                p_3 = list(p_3.cpu().detach().numpy()[0])
                p_4 = list(p_4.cpu().detach().numpy()[0])
                p_5 = list(p_5.cpu().detach().numpy()[0])
                p_6 = list(p_6.cpu().detach().numpy()[0])
                p_7 = list(p_7.cpu().detach().numpy()[0])

                predict_list = [p_1[0], p_2[0], p_3[0], p_4[0], p_5[0], p_6[0], p_7[0]]
                validlines_label.append(label_list)
                validlines_prediction.append(predict_list)

                # _, predicted = torch.max(predict.data, 1)
                # corrected = (predicted == label).squeeze()
                # corrected=corrected.detach().cpu().numpy()




                for i in range(batch_size):
                #     # lbl = class_correct[i]
                #     if corrected == True:
                #         class_correct[label.detach().cpu().numpy().squeeze()] += 1
                #         all_class_correct += 1
                    label_prob = label[i].detach().cpu().numpy().squeeze()
                    label_prob = np.argmax(label_prob)

                    class_total[int(label_prob)] += 1
                    # class_total[label.detach().cpu().numpy().squeeze()] += 1

                    # class_predict[predicted.detach().cpu().numpy().squeeze()]+=1
                    # all_class_total += 1


                # file.write(str(int(predicted.data)))
                # file.write('\n')
        # file.close()
        end = time.time()
        # print('validation:\tspend ' + str(end - start) + 's')

        # for index, (indv, total) in enumerate(zip(class_correct,class_total)):
        #     if class_total[index] !=0:
        #         class_accuracy[index]=str(class_correct[index] / class_total[index])
        #     else:
        #         class_accuracy[index] = str(-class_correct[index])

        cols=EMOTIONS

        validlines_label=np.array(validlines_label)
        validlines_prediction=np.array(validlines_prediction)
        try:
            df_label = pd.DataFrame(np.array(validlines_label), columns=EMOTIONS)
        except Exception as e:
            print("Unexpected error in creating dataframe:", e)

        try:
              df_prediction = pd.DataFrame(np.array(validlines_prediction), columns=EMOTIONS)
        except Exception as e:
            print("Unexpected error in creating dataframe:", e)





        df_label = df_label.rename(columns={x: cols[x] for x in range(len(cols))})
        df_prediction = df_prediction.rename(columns={x: cols[x] for x in range(len(cols))})



        compare_results=compare(df_label, df_prediction, emotions=EMOTIONS)


        # class_total=[str(clt) for clt in class_total]
        # class_predict=[str(prd) for prd in class_predict]
        #
        # class_accuracy = [str(round(float(item),3)) for item in class_accuracy]
        # class_predict = [str(int(float(item))) for item in class_predict]
        # class_total = [str(int(float(item))) for item in class_total]
        #
        #
        # enumerated_acc = ',  '.join(class_accuracy)
        # enumerated_predict = ','.join(class_predict)
        # enumerated_total = ','.join(class_total)

        try:
            print('validation - loss: ' + '{:3.6f}'.format(float(total_loss)/total_num))
            max_mean = 0.05
            max_std = 0.1
            print("max_mean : ", max_mean)
            print("max_std : ", max_std)
            for emo in EMOTIONS:
                mean_of_diff = compare_results[emo]['mean']
                std_of_diff = compare_results[emo]['std']

                success = 'PASS' if mean_of_diff < max_mean and std_of_diff < max_std else 'FAIL'
                print(' emotion {}, mean : {} + stdv :{} eval : {} class_total : {}'.format(emo,mean_of_diff,std_of_diff,success,class_total))
        except Exception as e:
            print(e)

    def test(self, test_path, epoch):
        start = time.time()
        self.model.eval()
        test_path += (str(epoch) + '.txt')
        file = open(test_path, 'w')
        with torch.no_grad():
            for batch_idx, (img) in enumerate(self.test_loader):
                img = img.to(self.device).float()
                _, weighted_prob, _ = self.model(img)

                _, predicted = torch.max(weighted_prob.data, 1)
                file.write(str(int(predicted.data)))
                file.write('\n')
        file.close()
        end = time.time()





        print('test:\tspend ' + str(end - start) + 's')

    def save_model(self, epoch, save_path='./model_save/resnet18_'):
        state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, save_path + str(epoch) + '.pth')

    def set_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr