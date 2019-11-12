import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_manager import ExperimentsManager
from sklearn.preprocessing import MinMaxScaler
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier
from device_classifier import DeviceClassifier
from multiple_device_classifier import MultipleDeviceClassifier
from sklearn import metrics

sns.set_style("white")

def eval_classifier(
        classifier,
        dataset,
        model_name,
        dataset_name,
        classification_method,
        seq_len,
        opt_seq_len,
        metrics_headers,
        metrics_dir):
    dataset_metrics, confusion_matrix = classifier.eval_on_dataset(dataset)
    shape = dataset_metrics['class'].shape
    dataset_metrics['device'] = np.full(shape, classifier.dev_name)
    dataset_metrics['model'] = np.full(shape, model_name)
    dataset_metrics['dataset'] = np.full(shape, dataset_name)
    dataset_metrics['classification_method'] = np.full(shape, classification_method)
    dataset_metrics['seq_len'] = np.full(shape, seq_len)
    dataset_metrics['opt_seq_len'] = np.full(shape, opt_seq_len)
    os.makedirs(metrics_dir, exist_ok=True)
    conf_matrix_dir = os.path.join(metrics_dir, 'confusion_matrix')
    os.makedirs(conf_matrix_dir, exist_ok=True)
    metrics_csv = os.path.join(metrics_dir, 'metrics.csv')
    confusion_matrix_csv = os.path.join(
        conf_matrix_dir, '{}_{}_{}_{}.csv'.format(model_name, dataset_name, 'session', seq_len))
    if os.path.exists(metrics_csv):
        header = False
    else:
        header = metrics_headers
    # dataset_metrics_df = pd.DataFrame({k: [v] for k, v in dataset_metrics.items()}, columns=metrics_headers)
    dataset_metrics_df = pd.DataFrame(dataset_metrics, columns=metrics_headers)
    dataset_metrics_df.to_csv(metrics_csv, mode='a', header=header, index=False)
    pd.DataFrame(confusion_matrix).to_csv(confusion_matrix_csv)


def run_experiment_with_datasets_devices(exp, datasets, devices, models_dir, metrics_dir):
    y_col = 'device_category'
    use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
    metrics_headers = list(pd.read_csv(os.path.abspath('data/metrics_headers.csv')))

    for dataset_name in datasets:
        print('@', dataset_name)
        dataset = utils.load_data_from_csv('data/{}.csv'.format(dataset_name), use_cols=use_cols)

        for dev_name in devices:
            print('@@', dev_name)
            for model_pkl in os.listdir(os.path.join(models_dir, dev_name)):
                model_name = os.path.splitext(model_pkl)[0]
                print('@@@', model_name)

                exp(y_col, use_cols, metrics_headers, models_dir, metrics_dir, dataset_name, dataset, dev_name, model_pkl, model_name)


def run_multi_dev_experiments(datasets, dev_model_csv, pred_methods):
    y_col = 'device_category'
    use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
    metrics_headers = list(pd.read_csv(os.path.abspath('data/metrics_headers.csv')))
    dev_model_combos = pd.read_csv(dev_model_csv)

    for dataset_name in datasets:
        print('@', dataset_name)
        dataset = utils.load_data_from_csv('data/{}.csv'.format(dataset_name), use_cols=use_cols)

        for idx, dev_model_combo in dev_model_combos.iterrows():
            print('@@', dev_model_combo)
            for pred_method in pred_methods:
                multi_dev_cls = MultipleDeviceClassifier(
                    dev_model_combo,
                    is_model_pkl=True,
                    pred_method=pred_method,
                    use_cols=use_cols,
                    y_col=y_col)

                eval_classifier(
                    classifier=multi_dev_cls,
                    dataset=dataset,
                    model_name=dev_model_combo.to_dict(),
                    dataset_name=dataset_name,
                    classification_method='multi_dev',
                    seq_len=dev_model_combo.to_dict(),
                    opt_seq_len=dev_model_combo.to_dict(),
                    metrics_headers=metrics_headers,
                    metrics_dir=metrics_dir)


def dev_sess_cls_exp(
        y_col,
        use_cols,
        metrics_headers,
        models_dir,
        metrics_dir,
        dataset_name,
        dataset,
        dev_name,
        model_pkl,
        model_name):
    # Device session classifier
    print("@@@@ Device session classifier")
    dev_sess_cls = DeviceSessionClassifier(
        dev_name,
        os.path.join(models_dir, dev_name, model_pkl),
        is_model_pkl=True,
        use_cols=use_cols,
        y_col=y_col)

    eval_classifier(
        classifier=dev_sess_cls,
        dataset=dataset,
        model_name=model_name,
        dataset_name=dataset_name,
        classification_method='session',
        seq_len=1,
        opt_seq_len=1,
        metrics_headers=metrics_headers,
        metrics_dir=metrics_dir)


def dev_seq_cls_exp(
        y_col,
        use_cols,
        metrics_headers,
        models_dir,
        metrics_dir,
        dataset_name,
        dataset,
        dev_name,
        model_pkl,
        model_name):
    # Device sequence classifier
    print("@@@@ Device sequence classifier")
    dev_seq_cls = DeviceSequenceClassifier(
        dev_name,
        os.path.join(models_dir, dev_name, model_pkl),
        is_model_pkl=True,
        use_cols=use_cols,
        y_col=y_col)

    if dataset_name == 'train':
        update = True
    else:
        update = False
    opt_seq_len = dev_seq_cls.find_opt_seq_len(dataset, update=update)
    print('{} seq_length: {}, optimal sequence length: {}'
          .format(dataset_name, dev_seq_cls.opt_seq_len, opt_seq_len))
    eval_classifier(
        classifier=dev_seq_cls,
        dataset=dataset,
        model_name=model_name,
        dataset_name=dataset_name,
        classification_method='sequence',
        seq_len=dev_seq_cls.opt_seq_len,
        opt_seq_len=opt_seq_len,
        metrics_headers=metrics_headers,
        metrics_dir=metrics_dir)

def test(dataset_csv, use_cols, device, dsc):
    validation = utils.load_data_from_csv(dataset_csv, use_cols=use_cols)

    all_sess = dsc.split_data(validation)[0]
    other_dev_sess = validation.groupby(dsc.y_col).get_group(device)
    other_dev_sess = dsc.split_data(other_dev_sess)[0]

    classification = 1 if device == device else 0

    # get the optimal sequence length for data
    opt_seq_len = dsc.find_opt_seq_len(validation)
    seqs = []
    for i in range(opt_seq_len):
        seqs.append(other_dev_sess[i])


    print("\n\n\npredicting all validation...")
    y_actual = [classification] * 99
    predicted = dsc.predict(other_dev_sess)
    print(metrics.accuracy_score(y_actual, predicted))


    print("\n\n\npredicting optimal...")
    y_actual = [classification] * len(seqs)
    predicted = dsc.predict(seqs)
    print(metrics.accuracy_score(y_actual, predicted))


use_cols = pd.read_csv(os.path.abspath("data/use_cols.csv"))

# test using their data
device = 'watch'
model_pkl = r'models/{0}/{0}_cart_entropy_100_samples_leaf.pkl'.format(device)
test('data/validation.csv', use_cols, 'watch', DeviceSequenceClassifier(device, model_pkl, is_model_pkl=True))

# test using our data
# COME HERE WHEN YOU HAVE DATA
