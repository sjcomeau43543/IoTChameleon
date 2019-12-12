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


    print("predicting all validation...")
    y_actual = [classification] * 99
    predicted = dsc.predict(other_dev_sess)
    print(metrics.accuracy_score(y_actual, predicted))


    print("predicting optimal...")
    y_actual = [classification] * len(seqs)
    predicted = dsc.predict(seqs)
    print(metrics.accuracy_score(y_actual, predicted))


def get_baseline_accuracies():

    use_cols = pd.read_csv(os.path.abspath("data/use_cols.csv"))
    dataset = 'data/validation.csv'

    for root,directory,files in os.walk('./models'):
        if './models' != root:
            print("testing", root)
            for model_pkl in files:
                print("model", model_pkl)
                classifier = DeviceSequenceClassifier(root, root+"/"+model_pkl, is_model_pkl=True)
                try:
                    test(dataset, use_cols, root.replace("./models","").replace("/",""), classifier)
                except:
                    continue

def get_data(dataset_csv, use_cols, device, dsc):
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

    # return sequences
    return seqs

def get_scores(predicted, actual):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for p, a in zip(predicted, actual):
        if p == a:
            if a == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if a == 1:
                fn = fn + 1
            else:
                fp = fp + 1

    print(tp, tn, fp, fn)

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = 0 if (tp+fp)==0 else tp / (tp + fp)
    reca = 0 if (tp+fn)==0 else tp / (tp + fn)
    f1sc = 0 if (prec+reca)==0 else 2*prec*reca / (prec + reca)

    print(acc, prec, reca, f1sc)


# get_baseline_accuracies()

# ----------
# 
classifnames = { 'baby_monitor'   : 'baby_monitor_cart_gini_200_samples_leaf',
                'lights'         : 'lights_cart_gini_200_samples_leaf',
                'motion_sensor'  : 'motion_sensor_cart_entropy_200_samples_leaf',
                'security_camera': 'security_camera_cart_entropy_200_samples_leaf',
                'smoke_detector' : 'smoke_detector_cart_entropy_200_samples_leaf',
                'socket'         : 'socket_cart_entropy_200_samples_leaf',
                # 'thermostat'     : '',
                # 'tv'             : 'tv_cart_gini_50_samples_leaf',
                'watch'          : 'watch_cart_entropy_100_samples_leaf',
                # 'water_sensor'   : 'water_sensor_cart_entropy_100_samples_leaf'
              }

classifiers = { 'baby_monitor'   : DeviceSequenceClassifier('./models', "./models/baby_monitor/"+classifnames['baby_monitor']+".pkl", is_model_pkl=True),
                'lights'         : DeviceSequenceClassifier('./models', "./models/lights/"+classifnames['lights']+".pkl", is_model_pkl=True),
                'motion_sensor'  : DeviceSequenceClassifier('./models', "./models/motion_sensor/"+classifnames['motion_sensor']+".pkl", is_model_pkl=True),
                'security_camera': DeviceSequenceClassifier('./models', "./models/security_camera/"+classifnames['security_camera']+".pkl", is_model_pkl=True),
                'smoke_detector' : DeviceSequenceClassifier('./models', "./models/smoke_detector/"+classifnames['smoke_detector']+".pkl", is_model_pkl=True),
                'socket'         : DeviceSequenceClassifier('./models', "./models/socket/"+classifnames['socket']+".pkl", is_model_pkl=True),
                # 'thermostat'     : '',
                # 'tv'             : DeviceSequenceClassifier('./models', "./models/tv/"+classifnames['tv']+".pkl", is_model_pkl=True),
                'watch'          : DeviceSequenceClassifier('./models', "./models/watch/"+classifnames['watch']+".pkl", is_model_pkl=True),
                # 'water_sensor'   : DeviceSequenceClassifier('./models', "./models/water_sensor/"+classifnames['water_sensor']+".pkl", is_model_pkl=True),
              }

data        = { 'baby_monitor'   : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'baby_monitor', classifiers['baby_monitor']),
                'lights'         : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'lights', classifiers['lights']),
                'motion_sensor'  : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'motion_sensor', classifiers['motion_sensor']),
                'security_camera': get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'security_camera', classifiers['security_camera']),
                'smoke_detector' : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'smoke_detector', classifiers['smoke_detector']),
                'socket'         : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'socket', classifiers['socket']),
                # 'thermostat'     : '',
                # 'tv'             : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'tv', classifiers['tv']),
                'watch'          : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'watch', classifiers['watch']),
                # 'water_sensor'   : get_data('data/validation.csv', pd.read_csv(os.path.abspath("data/use_cols.csv")), 'water_sensor', classifiers['water_sensor']),
              }

results     = { 'baby_monitor'   : [],
                'lights'         : [],
                'motion_sensor'  : [],
                'security_camera': [],
                'smoke_detector' : [],
                'socket'         : [],
                # 'thermostat'     : [],
                # 'tv'             : [],
                'watch'          : [],
                # 'water_sensor'   : [],
              }

for device in classifiers.keys():
    print("getting results for", device)
    r = results[device]
    p = []
    a = []

    modeldev = classifiers[device]

    for device2 in classifiers.keys():
        datadev2 = data[device2]
        print("\tcomparing to", device2, len(datadev2))

        r2 = 0
        for d in datadev2:
            e = classifiers[device].model.predict(d)
            e = e[0]
            
            if e == 1:
                r2 = r2+1

            # for later
            if device2 == device:
                a.append(1)
            else:
                a.append(0)
            p.append(e)
        r.append(r2)

    get_scores(p, a)

print("Printing results for confusion")
print(results)


# get_scores([0,0,1,1,1,1,1,1,1,1,0], [0,0,0,0,0,0,0,0,0,0,1])

