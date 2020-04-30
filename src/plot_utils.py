from os.path import join
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_plot_rc():
    '''
    Sets matplotlib rcParams to be similar to latex font
    Parameters:
        - None
    Returns:
        - None
    '''
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('text', usetex=True)


def autolabel(rects, ax):
    '''Attaches a text label above each bar in rects, displaying its height
    Parameters:
        - rects -- bars, matplotlib.container.BarContainer
        - ax -- axis, matplotlib.axes._subplots.AxesSubplot
    Returns:
        - None
    '''
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    rotation=90,
                    fontsize=12)


def plot_confusion_matrix(cm, ticks, labels, model_dir, model_name, features_type, checkpoint_epoch,
        verbose, mode, normalise=False):
    '''
    Plots a confusion matrix
    Parameters:
        - cm -- confusion matrix, numpy.ndarray
        - ticks -- ticks for xticks and yticks, numpy.ndarray or list
        - labels - list of audio labels, list or numpy.ndarray
        - model_dir -- path to saved model, str
        - model_name -- name of model, str
        - features_type -- method used to extract features, str
        - checkpoint_epoch -- epoch from which to use saved model, int
        - verbose -- if True, print the output to console, bool
        - mode -- either 'validation' or 'test', str
        - normalise -- whether normalise the confusion matrix, bool. Defaults to False
    Returns:
        - None
    '''
    (prefix, title_begin) = ('val', 'Validation') if mode == 'validation' else ('test', 'Test')

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    ax.set_title('{} {}confusion matrix for {} model using {} features'.\
            format(title_begin, 'normalised ' if normalise else '', model_name,
                features_type.replace('_', '-')), fontsize=16)
    mat = ax.matshow(cm, cmap='cividis')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mat, cax=cax)
    
    fig.tight_layout(pad=1.0)
    cm_path = join(model_dir, '{}_cm{}_at_epoch_{}.pdf'.format(prefix, '_norm' * normalise,
            checkpoint_epoch))
    fig.savefig(cm_path)
    if verbose:
        print('|{}onfusion matrix figure saved to\n|{}'.format('Normalised c' if normalise else 'C',
                cm_path))


def plot_train(params, validated, manually_verified_only, verbose):
    '''
    Function for plotting results of training: accuracies and losses
    Arguments:
        - params -- model parameters, hparams.HParamsFromYAML
        - validated -- where the model was trained on full train dataset or not, bool
        - manually_verified_only -- test model if it was trained using only manually verified audios
            for evaluation, bool
        - verbose -- if True, print the output to console, bool
    Returns:
        - None
    '''
    set_plot_rc()
    BASE_PATH = join(params.storage_dir, 'models', params.model_name, params.features_type)
    subbase_path = join('train_validation', 'holdout_fold={}'.format(params.holdout_fold)) \
            if validated else 'train_only'
    subbase_path = join(subbase_path, 'manually_verified_only') if manually_verified_only \
            else join(subbase_path, 'all')
    MODEL_DIR = join(BASE_PATH, subbase_path)
    
    raw_df = pd.read_json(join(MODEL_DIR, 'data.json'), orient='row')
    
    data_df = pd.DataFrame()
    data_df['epoch'] = raw_df['train']['epoch']
    
    for name, data in raw_df['train'].iteritems():
        if not name == 'epoch':
            data_df['{}_t'.format(name)] = data
    
    if validated:
        for name, data in raw_df['validation'].iteritems():
            if not name == 'epoch':
                data_df['{}_v'.format(name)] = data
        
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 9))
    
    ax1.set_title('Loss over training for {} model using {} features'.\
            format(params.model_name, params.features_type.replace('_', '-')), fontsize=18)
    ax1.plot(data_df['epoch'], data_df['loss_t'], '-o', color='blue', label='Training loss')
    if validated:
        ax1.plot(data_df['epoch'], data_df['loss_v'], '-d', color='orange', label='Validation loss')
    ax1.set_xticks(data_df['epoch'])
    ax1.set_xlabel('epoch', fontsize=16)
    ax1.set_ylabel('loss', fontsize=16)
    ax1.legend()
    
    ax2.set_title('Accuracy over training for {} model using {} features'.\
            format(params.model_name, params.features_type.replace('_', '-')), fontsize=18)
    ax2.plot(data_df['epoch'], data_df['accuracy_t'], '-o', color='green', label='Training accuracy')
    if validated:
        ax2.plot(data_df['epoch'], data_df['accuracy_v'], '-d', color='red', label='Validation accuracy')
    ax2.set_xticks(data_df['epoch'])
    ax2.set_xlabel('epoch', fontsize=16)
    ax2.set_ylabel('accuracy', fontsize=16)
    ax2.legend()
    
    ax3.set_title('Weighted F1-score over training for {} model using {} features'.\
            format(params.model_name, params.features_type.replace('_', '-')), fontsize=18)
    ax3.plot(data_df['epoch'], data_df['f1_score_t'], '-o', color='black', label='Training F1-score')
    if validated:
        ax3.plot(data_df['epoch'], data_df['f1_score_v'], '-d', color='magenta',
                label='Validation F1-score')
    ax3.set_xticks(data_df['epoch'])
    ax3.set_xlabel('epoch', fontsize=16)
    ax3.set_ylabel('F1-score', fontsize=16)
    ax3.legend()
    
    fig.tight_layout(pad=1.0)
    file_path = join(MODEL_DIR, 'data.pdf')
    fig.savefig(file_path)
    if verbose:
        print('|Figure saved to\n|{}'.format(file_path))


def plot_validation_test(params, checkpoint_epoch, validated, manually_verified_only, verbose,
        mode):
    '''
    Function for plotting results of validation: accuracies and confusion matrix
    Arguments:
        - params -- model parameters, hparams.HParamsFromYAML
        - checkpoint_epoch -- epoch from which to use saved model, int
        - validated -- where the model was trained on full train dataset or not, bool
        - manually_verified_only -- test model if it was trained using only manually verified audios
            for evaluation, bool
        - verbose -- if True, print the output to console, bool
        - mode -- either 'validation' or 'test', str
    Returns:
        - None
    '''
    set_plot_rc()

    (prefix, title_begin) = ('val', 'Validation') if mode == 'validation' else ('test', 'Test')

    BASE_PATH = join(params.storage_dir, 'models', params.model_name, params.features_type)
    subbase_path = join('train_validation', 'holdout_fold={}'.format(params.holdout_fold)) \
            if validated else 'train_only'
    subbase_path = join(subbase_path, 'manually_verified_only') if manually_verified_only \
            else join(subbase_path, 'all')
    MODEL_DIR = join(BASE_PATH, subbase_path)
    SCORES_PATH = join(MODEL_DIR, '{}_scores_at_epoch_{}.csv'.format(prefix, checkpoint_epoch))
    CONF_MAT_PATH = join(MODEL_DIR, '{}_cm_at_epoch_{}.csv'.format(prefix, checkpoint_epoch))
    
    # plotting scores
    scores_df = pd.read_csv(SCORES_PATH)
    LABELS = [s.replace('_', ' ') for s in scores_df['label']]
    x = np.arange(params.classes_number)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    ax1.set_title('{} F1-scores per class for {} model using {} features'.\
            format(title_begin, params.model_name, params.features_type.replace('_', '-')), fontsize=18)
    rects1 = ax1.bar(x, scores_df['F1_score'], width=0.8, edgecolor='black', alpha=0.7)
    autolabel(rects1, ax1)
    ax1.set_ylabel('F1-score', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=90)
    ax1.set_ylim([0, 1.25])
    ax1.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=14)
    
    ax2.set_title('{} accuracy scores per class for {} model using {} features'.\
            format(title_begin, params.model_name, params.features_type.replace('_', '-')), fontsize=18)
    rects2 = ax2.bar(x, scores_df['acc_score'], width=0.8, edgecolor='black', alpha=0.7)
    autolabel(rects2, ax2)
    ax2.set_ylabel('Accuracy score', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, rotation=90, fontsize=12)
    ax2.set_ylim([0, 1.25])
    ax2.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=14)
    
    fig.tight_layout(pad=1.0)
    scores_path = join(MODEL_DIR, '{}_scores_at_epoch_{}.pdf'.format(prefix, checkpoint_epoch)) 
    fig.savefig(scores_path)
    if verbose:
        print('|Scores figure saved to\n|{}'.format(scores_path))
    
    # plotting confusion matrix
    conf_mat_df = pd.read_csv(CONF_MAT_PATH)
    cm = conf_mat_df.values[:,1:]
    cm_norm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]   # normalise
    
    plot_confusion_matrix(
            cm=cm,
            ticks=x,
            labels=LABELS,
            model_dir=MODEL_DIR,
            model_name=params.model_name,
            features_type=params.features_type,
            checkpoint_epoch=checkpoint_epoch,
            verbose=verbose,
            mode=mode,
            normalise=False
            )
    plot_confusion_matrix(
            cm=cm_norm,
            ticks=x,
            labels=LABELS,
            model_dir=MODEL_DIR,
            model_name=params.model_name,
            features_type=params.features_type,
            checkpoint_epoch=checkpoint_epoch,
            verbose=verbose,
            mode=mode,
            normalise=True
            )
