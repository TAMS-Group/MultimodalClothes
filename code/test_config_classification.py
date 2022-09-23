"""
Author: Benny
Date: Nov 2019
"""
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import yaml
import json

from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def test(model, loader, config_dict, num_class=40):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    predictions = torch.tensor([]).cpu()
    targets = torch.tensor([]).cpu()

    for j, (data, target) in tqdm(enumerate(loader), total=len(loader)):
        if not config_dict['use_cpu']:
            target = target.cuda()

        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(config_dict['num_votes']):
            pred, _ = classifier(data)
            vote_pool += pred
        pred = vote_pool / config_dict['num_votes']
        pred_choice = pred.data.max(1)[1]
        predictions = torch.cat((predictions, pred_choice.cpu()), 0)
        targets = torch.cat((targets, target.cpu()), 0)
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
    labels = list(range(num_class))
    report = classification_report(targets, predictions, labels=labels)
    conf_matrix = confusion_matrix(targets, predictions, labels=labels)
    cm = confusion_matrix(targets, predictions, labels=labels, normalize='true')
    print(report)
    print(conf_matrix)
    if config_dict['tex_out']:
        class_names = config_dict['class_names']
        report_dict = classification_report(targets, predictions, labels=labels, output_dict=True)
        acc_str = '''\\begin{table}[]
		\\caption{}
		\\label{tab:}
		\\centering
		\\begin{tabular}{l|r|r|r|r}
			         & Precision & Recall & F1-score & Support <rows> \\\\\\hline\\hline
			Accuracy &           &        &     <acc> &     <supp> \\\\ \\hline
			Average & <avg_precision> & <avg_recall> & <avg_fscore> &     <supp> \\\\ \\hline
			Weighted Average & <wavg_precision> & <wavg_recall> & <wavg_fscore> &     <supp> \\\\ \\hline
		\\end{tabular}
	\\end{table}'''
        rows = ''
        for i, class_name in enumerate(class_names):
            rows += f'\\\\\\hline\n{class_name} & {report_dict[str(i)]["precision"]:.3f} & {report_dict[str(i)]["recall"]:.3f} & {report_dict[str(i)]["f1-score"]:.3f} & {report_dict[str(i)]["support"]}'
        acc_str = acc_str.replace('<rows>', rows)
        acc_str = acc_str.replace('<acc>', str(round(report_dict['accuracy'], 3)))
        acc_str = acc_str.replace('<supp>', str(round(report_dict['macro avg']['support'], 3)))
        acc_str = acc_str.replace('<avg_precision>', str(round(report_dict['macro avg']['precision'], 3)))
        acc_str = acc_str.replace('<avg_recall>', str(round(report_dict['macro avg']['recall'], 3)))
        acc_str = acc_str.replace('<avg_fscore>', str(round(report_dict['macro avg']['f1-score'], 3)))
        acc_str = acc_str.replace('<wavg_precision>', str(round(report_dict['weighted avg']['precision'], 3)))
        acc_str = acc_str.replace('<wavg_recall>', str(round(report_dict['weighted avg']['recall'], 3)))
        acc_str = acc_str.replace('<wavg_fscore>', str(round(report_dict['weighted avg']['f1-score'], 3)))

        conf_mat_str = '''
        \\begin{table}[t!]
            \\caption{}
            \\label{tab:}
            \\renewcommand{\\arraystretch}{1.5}
            \\setlength\\tabcolsep{0.08cm}
            \\centering
            \\begin{tabular}{|c|''' + "c|" * len(class_names) + '''}
                \\cline{2-''' + str(len(class_names) + 1) + '''}
                \\multicolumn{1}{c|}{} & \\rule{0pt}{17mm} <classes>\\\\\\hline
                <class_scores>
            \\end{tabular}
        \\end{table}'''
        classes_str = " & ".join(["\\rot{" + name + "}" for name in class_names])
        class_scores = '\n'.join([name + " & " + " & ".join(['\cellcolor{blue!' + str(int(cm[i][j] * 100)) + '}{' + str(round(cm[i][j],3)) + '}' for j in range(len(class_names))]) + '\\\\\\hline' for i, name in enumerate(class_names)])
        conf_mat_str = conf_mat_str.replace('<classes>', classes_str)
        conf_mat_str = conf_mat_str.replace('<class_scores>', class_scores)
        with open(config_dict['log_dir'] + '_eval.tex', 'w') as f:
            f.write(acc_str + '\n\n\n' + conf_mat_str)

    with open(config_dict['log_dir'] + '_eval.txt', 'w') as f:
        f.writelines([str(report), ' ', str(conf_matrix), ''])
    return instance_acc, class_acc


def main():
    def log_string(str):
        logger.info(str)
        print(str)

    with open(sys.argv[1], 'r') as f:
        config_dict = yaml.safe_load(f)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict['gpu'])

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + config_dict['log_dir']

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(json.dumps(config_dict))

    '''DATA LOADING'''
    log_string('Load dataset ...')

    modalities = list()
    if 'RGB' in config_dict['modalities']:
        modalities.append(MultiClothesDataLoader.Modalities.RGB)
    if 'DEPTH' in config_dict['modalities']:
        modalities.append(MultiClothesDataLoader.Modalities.DEPTH)
    if 'POINT_CLOUD' in config_dict['modalities']:
        modalities.append(MultiClothesDataLoader.Modalities.POINT_CLOUD)
    test_dataset = MultiClothesDataLoader(
            root=config_dict['data_path'],
            samples=config_dict['samples'],
            sample_mappings=config_dict['sample_mappings'],
            modalities=modalities,
            small_data=config_dict['small_data'],
            point_cloud_num_points=config_dict['num_points'])
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=config_dict['batch_size'], shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    class_names = config_dict['class_names']
    num_class = test_dataset.get_num_classes()
    if config_dict['tex_out']:
        if len(class_names) != num_class:
            print('class names and class count mismatch!')
            print(num_class)
            print(class_names)
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, config_dict)
    if not config_dict['use_cpu']:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, config_dict=config_dict, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    main()
