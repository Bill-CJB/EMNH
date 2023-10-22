####################################
# EXTERNAL LIBRARY
####################################
import torch
import torch.optim as optim
from torch import nn
import os
import shutil
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from tensorboard_logger import Logger as TbLogger

####################################
# INTERNAL LIBRARY
####################################
from source.utilities import Get_Logger, Average_Meter, augment_xy_data_by_64_fold_2obj
from matplotlib import pyplot as plt
import hvwfg

####################################
# PROJECT VARIABLES
####################################
from HYPER_PARAMS import *
from TORCH_OBJECTS import *



####################################
# PROJECT MODULES (to swap as needed)
####################################
import source.MODEL__Actor.grouped_actors as A_Module
import source.MODEL__Actor.grouped_actors_plk as A_Module_PLK
from source.mo_travelling_saleman_problem import TSP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT

if USE_CUDA:
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    device = torch.device('cuda', CUDA_DEVICE_NUM)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

torch.manual_seed(SEED)
np.random.seed(SEED)

############################################################################################################
############################################################################################################

# Objects to Use
actor = A_Module.ACTOR().to(device)
# optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
actor_plk = A_Module_PLK.ACTOR().to(device)
# optimizer_plk = optim.Adam(actor_plk.parameters(), lr=ACTOR_LEARNING_RATE)

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, model, modelpl):

        super(Meta, self).__init__()
        self.meta_lr = META_LR
        self.task_num = TASK_NUM

        self.model = deepcopy(model)
        self.modelpl = deepcopy(modelpl)
        self.optimizerpl = optim.Adam(self.modelpl.parameters(), lr=ACTOR_LEARNING_RATE)

    def forward(self, meta_lr=None, f=None):
        if f is None:
            f = torch.tensor([1., 1.])
        support_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * UPDATE_STEP, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
        self.meta_lr = meta_lr if meta_lr is not None else 0.1
        m_dict = deepcopy(self.model.state_dict())
        mpl_dict = deepcopy(self.modelpl.state_dict())
        k_name = ['node_prob_calculator.Wk_logit.weight']
        k_dict = deepcopy({name: m_dict[name] for name in k_name})
        for name in m_dict:
            if name in k_name:
                if m_dict[name].ndim == 2:
                    mpl_dict[name] = m_dict[name].repeat(self.task_num, 1)
                elif m_dict[name].ndim == 1:
                    mpl_dict[name] = m_dict[name].repeat(self.task_num)
            else:
                mpl_dict[name] = m_dict[name]
        self.modelpl.load_state_dict(mpl_dict)

        ww = torch.empty(self.task_num, 1, 1, 2)
        task_pair_num = self.task_num // 2
        alpha = 1
        for task_id in range(task_pair_num):
            random_num = np.random.dirichlet((alpha, alpha), None)
            ww[task_id, 0, 0] = torch.tensor(random_num)
        for task_id in range(task_pair_num, task_pair_num * 2):
            for j in range(2):
                ww[task_id, 0, 0, 1 - j] = ww[task_id - task_pair_num, 0, 0, j] * f[j] / f[1 - j]
            ww[task_id] = ww[task_id] / ww[task_id].sum(dim=-1, keepdim=True)
        for task_id in range(task_pair_num * 2, self.task_num):
            random_num = np.random.dirichlet((alpha, alpha), None)
            ww[task_id, 0, 0] = torch.tensor(random_num)
        print(ww.data)
        plk_dict_new = []
        for task_id in range(self.task_num):
            plk_dict_new.append(deepcopy(k_dict))

        self.modelpl.train()
        for data in tqdm(support_loader):
            # parallel data
            data = data.repeat(self.task_num, 1, 1)
            # data.shape = (batch_s, TSP_SIZE, 2)
            batch_s = data.size(0)

            # Actor Group Move
            ###############################################
            env = GROUP_ENVIRONMENT(data)
            group_s = TSP_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            self.modelpl.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                self.modelpl.update(group_state)
                action_probs = self.modelpl.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s,
                                                                                                           group_s)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch, group)
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # LEARNING - Actor
            ###############################################
            assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
            # reward was negative, here we set it to positive to calculate TCH
            reward = - reward
            reward = reward.reshape(self.task_num, TRAIN_BATCH_SIZE, TSP_SIZE, 2)
            if AGG == 1:
                ws_reward = (ww * reward).sum(dim=-1)
                agg_reward = ws_reward
            elif AGG == 2:
                z = torch.ones(reward.shape).cuda() * 0.0
                tch_reward = ww * (reward - z)
                tch_reward, _ = tch_reward.max(dim=-1)

                agg_reward = tch_reward
            agg_reward = agg_reward.reshape(self.task_num * TRAIN_BATCH_SIZE, TSP_SIZE)
            # set back reward to negative
            # reward = -reward
            agg_reward = -agg_reward

            group_reward = agg_reward
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch, group)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob
            # shape = (batch, group)
            group_loss = group_loss.reshape(self.task_num, TRAIN_BATCH_SIZE * TSP_SIZE)
            mhead_loss = group_loss.mean(dim=1)
            loss = mhead_loss.mean()

            self.optimizerpl.zero_grad()
            loss.backward()
            self.optimizerpl.step()

            for task_id in range(self.task_num):
                plk_dict_new[task_id] = {name: self.modelpl.state_dict()[name].chunk(self.task_num, dim=0)[task_id] for name in k_name}


        k_fw = {name: plk_dict_new[0][name] / float(self.task_num) for name in plk_dict_new[0]}
        for i in range(1, self.task_num):
            for name in plk_dict_new[i]:
                k_fw[name] += plk_dict_new[i][name] / float(self.task_num)

        m_dict = deepcopy(self.model.state_dict())
        mpl_dict = deepcopy(self.modelpl.state_dict())
        for name in m_dict:
            if name in k_name:
                m_dict[name] = k_dict[name] + (k_fw[name] - k_dict[name]) * self.meta_lr
            else:
                m_dict[name] = mpl_dict[name]
        self.model.load_state_dict(m_dict)

        return

    def finetune(self, weight=None, finetune_loader=None, model=None, fr=ACTOR_LEARNING_RATE):
        if weight is None:
            weight = torch.tensor([0.5, 0.5])
        if finetune_loader is None:
            finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
        if model is None:
            fine_model = deepcopy(self.model)
        else:
            fine_model = deepcopy(model)
        print("--------------------------------------------")
        print("weight:{}, {}".format(weight[0].item(), weight[1].item()))
        fine_model.train()
        fine_optimizer = optim.Adam(fine_model.parameters(), lr=fr)
        step = 0
        for data in finetune_loader:
            step += 1
            # data.shape = (batch_s, TSP_SIZE, 2)
            batch_s = data.size(0)

            # Actor Group Move
            ###############################################
            env = GROUP_ENVIRONMENT(data)
            group_s = TSP_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            fine_model.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                fine_model.update(group_state)
                action_probs = fine_model.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s,
                                                                                                           group_s)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch, group)
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # LEARNING - Actor
            ###############################################
            # reward was negative, here we set it to positive to calculate
            reward = - reward
            assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
            if AGG == 1:
                ws_reward = (weight * reward).sum(dim=2)
                agg_reward = ws_reward
            elif AGG == 2:
                z = torch.ones(reward.shape).cuda() * 0.0
                tch_reward = weight * (reward - z)
                tch_reward, _ = tch_reward.max(dim=2)
                agg_reward = tch_reward

            # set back reward to negative
            # reward = -reward
            agg_reward = -agg_reward

            group_reward = agg_reward
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch, group)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob
            # shape = (batch, group)
            loss = group_loss.mean()

            fine_optimizer.zero_grad()
            loss.backward()
            fine_optimizer.step()

            agg_reward = -agg_reward
            agg_obj, _ = agg_reward.min(dim=1)
            print('finetune_step:{}, agg_obj:{}, loss:{}'.format(step, agg_obj.mean().item(), loss))


        return fine_model

    def validate(self, model=None, validate_loader=None, weight=None):
        if model is None:
            valid_model = deepcopy(self.model)
        else:
            valid_model = model
        if validate_loader is None:
            validate_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
        if weight is None:
            weight = torch.tensor([1.0, 0.0])
        print("validating meta_model...")

        agg_obj = Average_Meter()
        obj0 = Average_Meter()
        obj1 = Average_Meter()
        valid_model.eval()
        for data in validate_loader:
            # data.shape = (batch_s, TSP_SIZE, 2)
            batch_s = data.size(0)

            # Actor Group Move
            ###############################################
            env = GROUP_ENVIRONMENT(data)
            group_s = TSP_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            valid_model.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                valid_model.update(group_state)
                action_probs = valid_model.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s,
                                                                                                           group_s)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch, group)
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # LEARNING - Actor
            ###############################################
            # reward was negative, here we set it to positive to calculate TCH
            # weight = torch.tensor([1.0, 0.0])
            reward = -reward
            assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
            if AGG == 1:
                ws_reward = (weight * reward).sum(dim=2)
                agg_reward = ws_reward
            elif AGG == 2:
                z = torch.ones(reward.shape).cuda() * 0.0
                tch_reward = weight * (reward - z)
                tch_reward, _ = tch_reward.max(dim=2)
                agg_reward = tch_reward
            # RECORDING
            ###############################################
            min_agg_obj, min_i = agg_reward.min(dim=1)
            agg_obj.push(min_agg_obj)
            min_i = min_i[:, None]
            min_obj0 = reward[:, :, 0].gather(1, min_i)
            min_obj1 = reward[:, :, 1].gather(1, min_i)
            obj0.push(min_obj0)
            obj1.push(min_obj1)
        agg_obj_avg = agg_obj.result()
        obj0_avg = obj0.result()
        obj1_avg = obj1.result()
        return [agg_obj_avg, obj0_avg, obj1_avg]

    def test(self, model=None, testdata=None, weight=None):
        if model is None:
            test_model = deepcopy(self.model)
        else:
            test_model = model
        if testdata is None:
            testdata = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 4))
        if weight is None:
            weight = torch.tensor([0.5, 0.5])
        print("testing meta_model...")

        test_model.eval()
        all_min_obj = torch.empty(TEST_DATASET_SIZE, 2)

        episode = 0
        while True:

            # seq = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 2))

            remaining = testdata.size(0) - episode
            batch_s = min(TEST_BATCH_SIZE, remaining)
            testdata_batch = testdata[episode: episode + batch_s]

            with torch.no_grad():

                env = GROUP_ENVIRONMENT(testdata_batch)
                group_s = TSP_SIZE
                group_state, reward, done = env.reset(group_size=group_s)
                test_model.reset(group_state)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    test_model.update(group_state)
                    action_probs = test_model.get_action_probabilities()
                    # shape = (batch, group, TSP_SIZE)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

                # reward was negative, here we set it to positive to calculate TCH
                reward = -reward
                assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
                if AGG == 1:
                    ws_reward = (weight * reward).sum(dim=2)
                    agg_reward = ws_reward
                elif AGG == 2:
                    z = torch.ones(reward.shape).cuda() * 0.0
                    tch_reward = weight * (reward - z)
                    tch_reward, _ = tch_reward.max(dim=2)
                    agg_reward = tch_reward
                # RECORDING
                ###############################################
                min_agg_obj, min_i = agg_reward.min(dim=1)
                min_i = min_i[:, None, None].repeat(1, 1, 2)
                min_obj = reward.gather(1, min_i)
                all_min_obj[episode: episode + batch_s] = torch.squeeze(min_obj, 1)

            episode = episode + batch_s
            if episode == TEST_DATASET_SIZE:
                break

        return all_min_obj

    def test_aug(self, model=None, testdata=None, weight=None):
        if model is None:
            test_model_aug = deepcopy(self.model)
        else:
            test_model_aug = model
        if testdata is None:
            testdata = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 4))
        if weight is None:
            weight = torch.tensor([0.5, 0.5])
        print("testing meta_model_aug...")

        test_model_aug.eval()
        all_min_obj = torch.empty(TEST_DATASET_SIZE, 2)

        testdata_aug = augment_xy_data_by_64_fold_2obj(testdata)
        testdata_aug = testdata_aug.reshape(64, -1, TSP_SIZE, 4).permute(1, 0, 2, 3).reshape(-1, TSP_SIZE, 4)

        episode = 0
        while True:

            # seq = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 2))

            remaining = testdata.size(0) - episode
            batch_s = min(TESTAUG_BATCH_SIZE, remaining) * 64
            testdata_batch = testdata_aug[episode * 64: episode * 64 + batch_s]

            with torch.no_grad():

                env = GROUP_ENVIRONMENT(testdata_batch)
                group_s = TSP_SIZE
                group_state, reward, done = env.reset(group_size=group_s)
                test_model_aug.reset(group_state)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    test_model_aug.update(group_state)
                    action_probs = test_model_aug.get_action_probabilities()
                    # shape = (batch, group, TSP_SIZE)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

                # reward was negative, here we set it to positive to calculate TCH
                reward = -reward
                assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
                if AGG == 1:
                    ws_reward = (weight * reward).sum(dim=2)
                    agg_reward = ws_reward
                elif AGG == 2:
                    z = torch.ones(reward.shape).cuda() * 0.0
                    tch_reward = weight * (reward - z)
                    tch_reward, _ = tch_reward.max(dim=2)
                    agg_reward = tch_reward
                # RECORDING
                ###############################################
                agg_reward = agg_reward.reshape(-1, 64 * group_s)
                reward = reward.reshape(-1, 64 * group_s, 2)

                min_agg_obj, min_i = agg_reward.min(dim=1)
                min_i = min_i[:, None, None].repeat(1, 1, 2)
                min_obj = reward.gather(1, min_i)
                all_min_obj[episode: episode + batch_s // 64] = torch.squeeze(min_obj, 1)

            episode = episode + batch_s // 64
            if episode == TEST_DATASET_SIZE:
                break

        return all_min_obj

# Meta Model
meta_learner = Meta(actor, actor_plk)

if MODE == 1:  # Train
    SAVE_FOLDER_NAME = 'TRAIN_' + METHOD + '_size{}'.format(TSP_SIZE)
    print(SAVE_FOLDER_NAME)

    # Make Log File
    # logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
    _, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    validate_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)

    tb_logger = TbLogger('logs/TSP_' + METHOD + '_n{}_{}'.format(TSP_SIZE, time.strftime("%Y%m%dT%H%M%S")))

    start_epoch = 0
    if LOAD_PATH is not None:
        # checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint_fullname = LOAD_PATH
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        actor.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Loaded trained_model")

    # save initial model
    checkpoint_dict = {
        'epoch': start_epoch,
        'model_state_dict': actor.state_dict()
    }
    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder_path, start_epoch))
    print("Saved meta_model")

    # GO
    f_avg = torch.tensor([1., 1.])
    weight = torch.tensor([0.5, 0.5])
    _, f0_avg, f1_avg = meta_learner.validate(weight=weight, validate_loader=validate_loader)
    f_avg[0] = f0_avg
    f_avg[1] = f1_avg
    for epoch in range(start_epoch, TOTAL_EPOCH + start_epoch):
        meta_lr = META_LR * (1. - epoch / float(TOTAL_EPOCH + start_epoch))
        meta_learner(meta_lr=meta_lr, f=f_avg)
        weight = torch.tensor([0.5, 0.5])
        _, f0_avg, f1_avg = meta_learner.validate(weight=weight, validate_loader=validate_loader)
        f_avg[0] = f0_avg
        f_avg[1] = f1_avg
        if ((epoch + 1) % (TOTAL_EPOCH // SAVE_NUM)) == 0:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': meta_learner.model.state_dict()
            }
            torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder_path, epoch + 1))
            print("Saved meta_model")

        print('Ep:{}({}%)  T:{}  meta_lr:{}'.format(epoch, epoch / TOTAL_EPOCH * 100,
                                                                   time.strftime("%H%M%S"), meta_lr))

elif MODE == 2:  # Finetune and Test
    print('TEST_' + METHOD + '_size{}'.format(TSP_SIZE))
    model_dir = MODEL_DIR
    n_weight = N_WEIGHT
    # testdata = Tensor(np.random.rand(TEST_DATASET_SIZE, TSP_SIZE, 4))
    testdata = torch.load('../test_tsp/testdata_tsp_size{}.pt'.format(TSP_SIZE))
    # testdata = torch.load('../test_tsp/KroAB{}.pt'.format(TSP_SIZE))
    testdata = testdata.to(device)
    test_save_ = 'test/' + METHOD + '_size{}-{}'.format(TSP_SIZE, time.strftime("%Y%m%d_%H%M"))
    # test_save_ = 'test/' + METHOD + '_KroAB{}-{}'.format(TSP_SIZE, time.strftime("%Y%m%d_%H%M"))
    weight = torch.zeros(2).cuda()
    if TSP_SIZE == 20:
        ref = np.array([20, 20])
    elif TSP_SIZE == 50:
        ref = np.array([35, 35])
    elif TSP_SIZE == 100:
        ref = np.array([65, 65])
    test_ep = TOTAL_EPOCH
    checkpoint_fullname = MODEL_DIR + '/checkpoint-{}.pt'.format(test_ep)
    test_save_dir = test_save_ + '/checkpoint-{}'.format(test_ep)
    os.makedirs(test_save_dir)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(test_save_dir)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    # start_epoch = checkpoint['epoch']
    print('Loaded meta-model-{}'.format(test_ep))
    sols = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    sols_aug = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    # dichotomy
    k = FINETUNE_STEP  # finetune step in each level
    hw = [[] for i in range(7)]
    model_list = [[] for i in range(7)]
    model_list[0].append(deepcopy(actor.state_dict()))
    for j in range(7):
        hw[j] = [[i / (2 ** j) + 1 / (2 ** j) / 2, 1 - i / (2 ** j) - 1 / (2 ** j) / 2] for i in range(2 ** j)]
    for j in range(1, 7):
        print('finetune_level:{}'.format(j))
        finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * k, num_nodes=TSP_SIZE,
                                                  batch_size=TRAIN_BATCH_SIZE)
        for wi, wv in enumerate(hw[j]):
            weight[0] = wv[0]
            weight[1] = wv[1]
            actor.load_state_dict(model_list[j - 1][wi // 2])
            fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
            model_list[j].append(deepcopy(fine_model.state_dict()))
    print('finetune_level_last')
    finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * k, num_nodes=TSP_SIZE,
                                              batch_size=TRAIN_BATCH_SIZE)
    for i in range(n_weight - 1):
        weight[0] = 0.01 * i
        weight[1] = 1 - 0.01 * i
        weight = weight / torch.sum(weight)
        actor.load_state_dict(model_list[6][i * (2 ** 6) // (n_weight - 1)])
        fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
        cost_obj = meta_learner.test(model=fine_model, testdata=testdata, weight=weight)
        cost_obj_aug = meta_learner.test_aug(model=fine_model, testdata=testdata, weight=weight)
        sols[i] = np.array(cost_obj.cpu())
        sols_aug[i] = np.array(cost_obj_aug.cpu())
    for i in range(n_weight - 1, n_weight):
        weight[0] = 0.01 * i
        weight[1] = 1 - 0.01 * i
        weight = weight / torch.sum(weight)
        actor.load_state_dict(model_list[6][-1])
        fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
        cost_obj = meta_learner.test(model=fine_model, testdata=testdata, weight=weight)
        cost_obj_aug = meta_learner.test_aug(model=fine_model, testdata=testdata, weight=weight)
        sols[i] = np.array(cost_obj.cpu())
        sols_aug[i] = np.array(cost_obj_aug.cpu())

    all_sols = sols.swapaxes(0, 1)
    all_sols_aug = sols_aug.swapaxes(0, 1)
    # calculate hv
    hv_ratio = np.empty((TEST_DATASET_SIZE))
    hv_ratio_aug = np.empty((TEST_DATASET_SIZE))
    os.makedirs(os.path.join(test_save_dir, "sols"))
    os.makedirs(os.path.join(test_save_dir, "sols_aug"))
    for i in range(TEST_DATASET_SIZE):
        np.savetxt(os.path.join(test_save_dir, "sols", "ins{}.txt".format(i)), all_sols[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        np.savetxt(os.path.join(test_save_dir, "sols_aug", "ins{}.txt".format(i)), all_sols_aug[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        hv = hvwfg.wfg(all_sols[i].astype(float), ref.astype(float))
        hv_aug = hvwfg.wfg(all_sols_aug[i].astype(float), ref.astype(float))
        hv_ratio[i] = hv / (ref[0] * ref[1])
        hv_ratio_aug[i] = hv_aug / (ref[0] * ref[1])
    np.savetxt(os.path.join(test_save_dir, "mean_sols.txt"), np.mean(all_sols, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "mean_sols_aug.txt"), np.mean(all_sols_aug, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    print(MODEL_DIR)
    print('meta-model-{}'.format(test_ep))
    print('HV Ratio: {:.4f}'.format(hv_ratio.mean()))
    print('HV Ratio Aug: {:.4f}'.format(hv_ratio_aug.mean()))
    np.savetxt(os.path.join(test_save_dir, "all_hv.txt"), hv_ratio, fmt='%1.4f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "all_hv_aug.txt"), hv_ratio_aug, fmt='%1.4f', delimiter='\t')
    file = open(test_save_ + '/hv.txt', 'w')
    file.write('mean_hv: ' + str(hv_ratio.mean()) + '\n')
    file.write('mean_hv_aug: ' + str(hv_ratio_aug.mean()) + '\n')


elif MODE == 3:  # Test all finetune from meta
    print('TEST_ALLFT_' + METHOD + '_size{}'.format(TSP_SIZE))
    model_dir = MODEL_DIR
    n_weight = N_WEIGHT
    # testdata = Tensor(np.random.rand(TEST_DATASET_SIZE, TSP_SIZE, 4))
    testdata = torch.load('../test_tsp/testdata_tsp_size{}.pt'.format(TSP_SIZE))
    testdata = testdata.to(device)
    test_save_ = 'test/' + METHOD + '_size{}-{}'.format(TSP_SIZE, time.strftime("%Y%m%d_%H%M"))
    weight = torch.zeros(2).cuda()
    if TSP_SIZE == 20:
        ref = np.array([20, 20])
    elif TSP_SIZE == 50:
        ref = np.array([35, 35])
    elif TSP_SIZE == 100:
        ref = np.array([65, 65])
    test_ep = TOTAL_EPOCH
    checkpoint_fullname = MODEL_DIR + '/checkpoint-{}.pt'.format(test_ep)
    test_save_dir = test_save_ + '/checkpoint-{}'.format(test_ep)
    os.makedirs(test_save_dir)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(test_save_dir)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    # start_epoch = checkpoint['epoch']
    print('Loaded meta-model-{}'.format(test_ep))
    finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_nodes=TSP_SIZE,
                                              batch_size=TRAIN_BATCH_SIZE)
    sols = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    sols_aug = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    run_time = time.time() - time.time()
    run_time_aug = time.time() - time.time()
    for i in range(n_weight):
        weight[0] = 1 - 0.01 * i
        weight[1] = 0.01 * i
        weight = weight / torch.sum(weight)
        fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
        timer_start = time.time()
        cost_obj = meta_learner.test(model=fine_model, testdata=testdata, weight=weight)
        run_time += time.time() - timer_start
        timer_start_aug = time.time()
        cost_obj_aug = meta_learner.test_aug(model=fine_model, testdata=testdata, weight=weight)
        run_time_aug += time.time() - timer_start_aug
        sols[i] = np.array(cost_obj.cpu())
        sols_aug[i] = np.array(cost_obj_aug.cpu())
    print('time: ', run_time)
    print('time_aug: ', run_time_aug)
    all_sols = sols.swapaxes(0, 1)
    all_sols_aug = sols_aug.swapaxes(0, 1)
    # calculate hv
    hv_ratio = np.empty((TEST_DATASET_SIZE))
    hv_ratio_aug = np.empty((TEST_DATASET_SIZE))
    os.makedirs(os.path.join(test_save_dir, "sols"))
    os.makedirs(os.path.join(test_save_dir, "sols_aug"))
    for i in range(TEST_DATASET_SIZE):
        np.savetxt(os.path.join(test_save_dir, "sols", "ins{}.txt".format(i)), all_sols[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        np.savetxt(os.path.join(test_save_dir, "sols_aug", "ins{}.txt".format(i)), all_sols_aug[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        hv = hvwfg.wfg(all_sols[i].astype(float), ref.astype(float))
        hv_aug = hvwfg.wfg(all_sols_aug[i].astype(float), ref.astype(float))
        hv_ratio[i] = hv / (ref[0] * ref[1])
        hv_ratio_aug[i] = hv_aug / (ref[0] * ref[1])
    np.savetxt(os.path.join(test_save_dir, "mean_sols.txt"), np.mean(all_sols, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "mean_sols_aug.txt"), np.mean(all_sols_aug, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    print(MODEL_DIR)
    print('meta-model-{}'.format(test_ep))
    print('HV Ratio: {:.4f}'.format(hv_ratio.mean()))
    print('HV Ratio Aug: {:.4f}'.format(hv_ratio_aug.mean()))
    np.savetxt(os.path.join(test_save_dir, "all_hv.txt"), hv_ratio, fmt='%1.4f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "all_hv_aug.txt"), hv_ratio_aug, fmt='%1.4f', delimiter='\t')
    file = open(test_save_ + '/hv.txt', 'w')
    file.write('mean_hv: ' + str(hv_ratio.mean()) + '\n')
    file.write('mean_hv_aug: ' + str(hv_ratio_aug.mean()) + '\n')


elif MODE == 4:  # Save Submodels
    print('Submodels_' + METHOD + '_size{}'.format(TSP_SIZE))
    model_dir = MODEL_DIR
    n_weight = N_WEIGHT
    save_ = 'submodel/size{}_FT{}'.format(TSP_SIZE, FINETUNE_STEP)
    weight = torch.zeros(2).cuda()
    test_ep = TOTAL_EPOCH
    checkpoint_fullname = MODEL_DIR + '/checkpoint-{}.pt'.format(test_ep)
    os.makedirs(save_)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(save_)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    # start_epoch = checkpoint['epoch']
    print('Loaded meta-model-{}'.format(test_ep))
    sols = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    sols_aug = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    # dichotomy
    k = FINETUNE_STEP  # finetune step in each level
    hw = [[] for i in range(7)]
    model_list = [[] for i in range(7)]
    model_list[0].append(deepcopy(actor.state_dict()))
    for j in range(7):
        hw[j] = [[i / (2 ** j) + 1 / (2 ** j) / 2, 1 - i / (2 ** j) - 1 / (2 ** j) / 2] for i in range(2 ** j)]
    for j in range(1, 7):
        print('finetune_level:{}'.format(j))
        finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * k, num_nodes=TSP_SIZE,
                                                  batch_size=TRAIN_BATCH_SIZE)
        for wi, wv in enumerate(hw[j]):
            weight[0] = wv[0]
            weight[1] = wv[1]
            actor.load_state_dict(model_list[j - 1][wi // 2])
            fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
            model_list[j].append(deepcopy(fine_model.state_dict()))
    print('finetune_level_last')
    finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * k, num_nodes=TSP_SIZE,
                                              batch_size=TRAIN_BATCH_SIZE)
    for i in range(n_weight - 1):
        weight[0] = 0.01 * i
        weight[1] = 1 - 0.01 * i
        weight = weight / torch.sum(weight)
        actor.load_state_dict(model_list[6][i * (2 ** 6) // (n_weight - 1)])
        fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
        checkpoint_dict = {
            'model_state_dict': fine_model.state_dict()
        }
        torch.save(checkpoint_dict, '{}/submodel-{}.pt'.format(save_, i))

    for i in range(n_weight - 1, n_weight):
        weight[0] = 0.01 * i
        weight[1] = 1 - 0.01 * i
        weight = weight / torch.sum(weight)
        actor.load_state_dict(model_list[6][-1])
        fine_model = meta_learner.finetune(weight=weight, finetune_loader=finetune_loader, model=actor)
        checkpoint_dict = {
            'model_state_dict': fine_model.state_dict()
        }
        torch.save(checkpoint_dict, '{}/submodel-{}.pt'.format(save_, i))
    print("Saved Submodels")

elif MODE == 5:  # Load Submodels to Test
    print('Load Submodels to Test_' + METHOD + '_size{}'.format(TSP_SIZE))
    n_weight = N_WEIGHT
    testdata = torch.load('../test_tsp/testdata_tsp_size{}.pt'.format(TSP_SIZE))
    # testdata = torch.load('../test_tsp/KroAB{}.pt'.format(TSP_SIZE))
    testdata = testdata.to(device)
    # test_save_ = 'test/KroAB{}_FT{}'.format(TSP_SIZE, FINETUNE_STEP)
    test_save_ = 'test/testdata_tsp_size{}_FT{}'.format(TSP_SIZE, FINETUNE_STEP)
    weight = torch.zeros(2).cuda()
    if TSP_SIZE == 20:
        ref = np.array([20, 20])
    elif TSP_SIZE == 50:
        ref = np.array([35, 35])
    elif TSP_SIZE == 100:
        ref = np.array([65, 65])
    elif TSP_SIZE == 150:
        ref = np.array([85, 85])   #150
    elif TSP_SIZE == 200:
        ref = np.array([115, 115])   #200
    test_save_dir = test_save_
    os.makedirs(test_save_dir)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(test_save_dir)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    sols = np.empty((n_weight, TEST_DATASET_SIZE, 2))
    sols_aug = np.empty((n_weight, TEST_DATASET_SIZE, 2))

    run_time = time.time() - time.time()
    run_time_aug = time.time() - time.time()

    for i in range(n_weight):
        weight[0] = 0.01 * i
        weight[1] = 1 - 0.01 * i
        weight = weight / torch.sum(weight)
        checkpoint = torch.load('submodel/size100_FT{}/submodel-{}.pt'.format(FINETUNE_STEP, i), map_location=device)
        actor.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded submodel-{}'.format(i))
        timer_start = time.time()
        cost_obj = meta_learner.test(model=actor, testdata=testdata, weight=weight)
        run_time += time.time() - timer_start
        timer_start_aug = time.time()
        cost_obj_aug = meta_learner.test_aug(model=actor, testdata=testdata, weight=weight)
        run_time_aug += time.time() - timer_start_aug
        sols[i] = np.array(cost_obj.cpu())
        sols_aug[i] = np.array(cost_obj_aug.cpu())

    print('time: ', run_time)
    print('time_aug: ', run_time_aug)

    all_sols = sols.swapaxes(0, 1)
    all_sols_aug = sols_aug.swapaxes(0, 1)
    # calculate hv
    hv_ratio = np.empty((TEST_DATASET_SIZE))
    hv_ratio_aug = np.empty((TEST_DATASET_SIZE))
    os.makedirs(os.path.join(test_save_dir, "sols"))
    os.makedirs(os.path.join(test_save_dir, "sols_aug"))
    for i in range(TEST_DATASET_SIZE):
        np.savetxt(os.path.join(test_save_dir, "sols", "ins{}.txt".format(i)), all_sols[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        np.savetxt(os.path.join(test_save_dir, "sols_aug", "ins{}.txt".format(i)), all_sols_aug[i],
                   fmt='%1.4f\t%1.4f', delimiter='\t')
        hv = hvwfg.wfg(all_sols[i].astype(float), ref.astype(float))
        hv_aug = hvwfg.wfg(all_sols_aug[i].astype(float), ref.astype(float))
        hv_ratio[i] = hv / (ref[0] * ref[1])
        hv_ratio_aug[i] = hv_aug / (ref[0] * ref[1])
    np.savetxt(os.path.join(test_save_dir, "mean_sols.txt"), np.mean(all_sols, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "mean_sols_aug.txt"), np.mean(all_sols_aug, axis=0), fmt='%1.8f\t%1.8f', delimiter='\t')
    print('HV Ratio: {:.4f}'.format(hv_ratio.mean()))
    print('HV Ratio Aug: {:.4f}'.format(hv_ratio_aug.mean()))
    np.savetxt(os.path.join(test_save_dir, "all_hv.txt"), hv_ratio, fmt='%1.4f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "all_hv_aug.txt"), hv_ratio_aug, fmt='%1.4f', delimiter='\t')
    file = open(test_save_ + '/hv.txt', 'w')
    file.write('mean_hv: ' + str(hv_ratio.mean()) + '\n')
    file.write('mean_hv_aug: ' + str(hv_ratio_aug.mean()) + '\n')

