import math
import os
import sys
import torch
import numpy as np

import datetime
import logging

import yaml
import code
import provider
import importlib
import shutil
import argparse
from typing import Optional, Dict, List, Union

from pathlib import Path
from tqdm import tqdm
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader

from torch.utils.data import DataLoader, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


class Trainer:

    def __init__(self, config_path: str, gpu: Optional[Union[int, str]] = None):

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(gpu)
        self.config_dict = config_dict
        if not gpu:
            gpu = self.config_dict['gpu']

        '''HYPER PARAMETER'''
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.out = True

        '''CREATE DIR'''
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self.exp_dir = Path('./log/')
        self.exp_dir.mkdir(exist_ok=True)
        self.exp_dir = self.exp_dir.joinpath('classification')
        self.exp_dir.mkdir(exist_ok=True)
        exp_dir = self.exp_dir.joinpath(timestr)
        exp_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = exp_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        log_dir = exp_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)

        '''LOG'''
        self.logger = logging.getLogger("Model")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, self.config_dict['model']))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.log_string('PARAMETER ...')
        self.log_string(yaml.dump(self.config_dict))





    def inplace_relu(self, m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    def load_data(self, batch_size=None, num_points=None):
        self.log_string('Load dataset ...')
        batch_size = self.config_dict['batch_size']
        modalities = list()
        if 'RGB' in self.config_dict['modalities']:
            modalities.append(MultiClothesDataLoader.Modalities.RGB)
        if 'DEPTH' in self.config_dict['modalities']:
            modalities.append(MultiClothesDataLoader.Modalities.DEPTH)
        if 'POINT_CLOUD' in self.config_dict['modalities']:
            modalities.append(MultiClothesDataLoader.Modalities.POINT_CLOUD)

        self.dataset = MultiClothesDataLoader(
            root=self.config_dict['data_path'],
            samples=self.config_dict['samples'],
            sample_mappings=self.config_dict['sample_mappings'],
            modalities=modalities,
            small_data=self.config_dict['small_data'],
            point_cloud_num_points=self.config_dict['num_points'],
        )

        n_val = int(len(self.dataset) * self.config_dict['validation_split'])
        n_train = len(self.dataset) - n_val
        n_val = math.ceil(n_val * self.config_dict['dataset_portion'])
        n_train = math.ceil(n_train * self.config_dict['dataset_portion'])
        ignore = len(self.dataset) - n_val - n_train
        self.train_dataset, self.validation_dataset, _ = torch.utils.data.random_split(self.dataset, [n_train, n_val, ignore], generator=torch.Generator().manual_seed(self.config_dict['random_seed']))
        self.trainDataLoader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.config_dict['data_loader_workers'], drop_last=True)
        self.validationDataLoader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False, num_workers=self.config_dict['data_loader_workers'])

    def test(self, model, num_class=40):
        mean_correct = []
        class_acc = np.zeros((num_class, 3))
        classifier = model.eval()

        for j, (data, target) in tqdm(enumerate(self.validationDataLoader), total=len(self.validationDataLoader)):
        # for j, (data, target) in enumerate(self.validationDataLoader):

            if not self.config_dict['use_cpu']:
                target = target.cuda()
            pred, _ = classifier(data)
            pred_choice = pred.data.max(1)[1]
            bs = len(data[list(data.keys())[0]])  # get batch size of the individual batch (in case it is the last one)

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(data[list(data.keys())[0]][target == cat].size()[0])
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(bs))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        return instance_acc, class_acc

    def log_string(self, str):
        if self.out:
            self.logger.info(str)
            print(str)

    def train_config(self):
        self.train(
            self.config_dict['batch_size'],
            self.config_dict['num_points'],
            self.config_dict['learning_rate'],
            self.config_dict['dgcnn_k'],
            self.config_dict['dropout'],
            self.config_dict['emb_dims'],
            self.config_dict['model'],
            self.config_dict['use_colors'],
            self.config_dict['decay_rate'],
            self.config_dict['sched_step_size'],
            self.config_dict['sched_gamma'],
        )

    def train_pointnet_optuna(self, trial):
        self.out = False
        num_points = trial.suggest_categorical('num_points', [256, 512, 1024, 2048, 4096])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.1)
        decay_rate = trial.suggest_loguniform('decay_rate', 1e-6, 1e-2)
        emb_dims = trial.suggest_int('emb_dims', 512, 4096)
        sched_step_size = trial.suggest_int('sched_step_size', 5, 50)
        sched_gamma = trial.suggest_float('sched_gamma', 0, 1.0 , step=0.05)
        return self.train(self.config_dict['batch_size'], num_points, learning_rate, 5, dropout, emb_dims, 'pointnet_cls', self.config_dict['use_colors'], decay_rate, sched_step_size, sched_gamma, trial=trial)

    def train_pointnetpp_optuna(self, trial):
        self.out = False
        num_points = trial.suggest_categorical('num_points', [256, 512, 1024, 2048, 4096])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.1)
        decay_rate = trial.suggest_loguniform('decay_rate', 1e-6, 1e-2)
        emb_dims = trial.suggest_int('emb_dims', 512, 4096)
        sched_step_size = trial.suggest_int('sched_step_size', 5, 50)
        sched_gamma = trial.suggest_float('sched_gamma', 0, 1.0 , step=0.05)
        return self.train(self.config_dict['batch_size'], num_points, learning_rate, 5, dropout, emb_dims, 'pointnet2_cls_msg', self.config_dict['use_colors'], decay_rate, sched_step_size, sched_gamma, trial=trial)

    def train_dgcnn_optuna(self, trial):
        self.out = False
        num_points = trial.suggest_categorical('num_points', [256, 512, 1024, 2048, 4096])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
        k = trial.suggest_int('k', 5, 50)
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.1)
        decay_rate = trial.suggest_loguniform('decay_rate', 1e-6, 1e-2)
        emb_dims = trial.suggest_int('emb_dims', 1024, 4096)
        sched_step_size = trial.suggest_int('sched_step_size', 5, 50)
        sched_gamma = trial.suggest_float('sched_gamma', 0, 1.0 , step=0.05)
        return self.train(
            self.config_dict['batch_size'],
            num_points,
            learning_rate,
            k,
            dropout,
            emb_dims,
            'dgcnn_cls',
            self.config_dict['use_colors'],
            decay_rate,
            sched_step_size,
            sched_gamma,
            trial=trial
        )

    def train(self, batch_size, num_points, learning_rate, k, dropout, emb_dims, modelname, use_color, decay_rate, sched_step_size, sched_gamma, trial=None):

        self.load_data(batch_size=batch_size, num_points=num_points)
        num_class = self.dataset.get_num_classes()
        if self.out:
            self.log_string('Classes:')
            self.log_string(num_class)

        '''MODEL LOADING'''
        model = importlib.import_module(modelname)
        shutil.copy('./models/%s.py' % modelname, str(self.exp_dir))
        shutil.copy('models/pointnet2_utils.py', str(self.exp_dir))
        shutil.copy('./train_dgcnn_classification.py', str(self.exp_dir))

        classifier = model.get_model(num_class=num_class, config_dict=self.config_dict)
        criterion = model.get_loss()
        classifier.apply(self.inplace_relu)

        if not self.config_dict['use_cpu']:
            classifier = classifier.cuda()
            criterion = criterion.cuda()

        try:
            checkpoint = torch.load(str(self.exp_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            if self.out:
                self.log_string('Use pretrain model')
        except:
            if self.out:
                self.log_string('No existing model, starting training from scratch...')
            start_epoch = 0

        if self.config_dict['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=float(decay_rate)
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)
        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0

        '''TRANING'''
        if self.out:
            self.logger.info('Start training...')
        for epoch in range(start_epoch, self.config_dict['epoch']):
            if self.out:
                self.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.config_dict['epoch']))
            mean_correct = []
            classifier = classifier.train()

            for batch_id, (data, target) in tqdm(enumerate(self.trainDataLoader, 0), total=len(self.trainDataLoader), smoothing=0.9):
            # for batch_id, (data, target) in enumerate(self.trainDataLoader, 0):
                optimizer.zero_grad()
                data: dict

                if not self.config_dict['use_cpu']:
                    target = target.cuda()

                pred, trans_feat = classifier(data)
                loss = criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]
                bs = len(data[list(data.keys())[0]])  # get batch size of the individual batch (in case it is the last one)

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(bs))
                loss.backward()
                optimizer.step()
                global_step += 1

            train_instance_acc = np.mean(mean_correct)
            if self.out:
                self.log_string('Train Instance Accuracy: %f' % train_instance_acc)

            with torch.no_grad():
                instance_acc, class_acc = self.test(classifier.eval(), num_class=num_class)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                if self.out:
                    self.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                    self.log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    savepath = str(self.checkpoints_dir) + '/best_model.pth'
                    if self.out:
                            self.logger.info('Save model...')
                            self.log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                global_epoch += 1
            if trial:
                trial.report(best_class_acc, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            scheduler.step()
        if self.out:
            self.log_string('End of training...')
        return best_class_acc


if __name__ == '__main__':
    t = Trainer(sys.argv[1], sys.argv[2])
    if '-od' in sys.argv:
        # with optuna
        import optuna
        # study = optuna.create_study(direction='maximize')
        study = optuna.load_study(study_name="dgcnn2", storage="postgresql://15fiedler:uMLtzMNz5EFzyMlQ5Y4EYV0y@rzrobo3/optuna_studies")
        study.optimize(t.train_dgcnn_optuna, timeout=24*60*60)
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result = study.trials_dataframe()
        fig = optuna.visualization.plot_param_importances(study)
        os.makedirs('results_d/images', exist_ok=True)
        fig.write_image("results_d/images/importance.png")
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("results_d/images/history.png")


        result.to_pickle("results_d/results.pkl")
        result.to_csv("results_d/results.csv")
    elif '-op' in sys.argv:
        # with optuna
        import optuna
        # study = optuna.create_study(direction='maximize')
        study = optuna.load_study(study_name="pointnet1", storage="postgresql://15fiedler:uMLtzMNz5EFzyMlQ5Y4EYV0y@rzrobo3/optuna_studies")
        study.optimize(t.train_pointnet_optuna, timeout=24*60*60)
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result = study.trials_dataframe()
        fig = optuna.visualization.plot_param_importances(study)
        os.makedirs('results_p/images', exist_ok=True)
        fig.write_image("results_p/images/importance.png")
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("results_p/images/history.png")


        result.to_pickle("results_p/results.pkl")
        result.to_csv("results_p/results.csv")

    elif '-op2' in sys.argv:
        # with optuna
        import optuna
        # study = optuna.create_study(direction='maximize')
        study = optuna.load_study(study_name="pointnetpp1", storage="postgresql://15fiedler:uMLtzMNz5EFzyMlQ5Y4EYV0y@rzrobo3/optuna_studies")
        study.optimize(t.train_pointnetpp_optuna, timeout=24*60*60)
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result = study.trials_dataframe()
        fig = optuna.visualization.plot_param_importances(study)
        os.makedirs('results_p++/images', exist_ok=True)
        fig.write_image("results_p++/images/importance.png")
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("results_p++/images/history.png")


        result.to_pickle("results_p++/results.pkl")
        result.to_csv("results_p++/results.csv")

    else:
        # without optuna
        t.train_config()
