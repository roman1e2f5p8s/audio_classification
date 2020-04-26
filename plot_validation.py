from os.path import join
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hparams import HParamsFromYAML


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


def plot_confusion_matrix(cm, normalise=False):
    '''
    Plots a confusion matrix
    Parameters:
        - cm -- confusion matrix, numpy.ndarray
        - normalise -- whether normalise the confusion matrix, bool. Defaults to False
    Returns:
        - None
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_title('{}onfusion matrix for {} model using the validation dataset'.\
            format('Normalised c' if normalise else 'C', args_model_name), fontsize=16)
    mat = ax.matshow(cm, cmap='cividis')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=90, fontsize=10)
    ax.set_yticks(x)
    ax.set_yticklabels(LABELS, rotation=0, fontsize=10)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mat, cax=cax)
    
    fig.tight_layout(pad=1.0)
    fig.savefig(join(MODEL_DIR, 'cm{}_at_epoch_{}.pdf'.format('_norm' * normalise,
        args_checkpoint_epoch)))


args_checkpoint_epoch = 10
args_model_name = 'VGGish'

params = HParamsFromYAML('hparams.yaml', param_set=args_model_name)

MODEL_DIR = join(params.storage_dir, 'models', args_model_name, 'train_validation',
    'holdout_fold={}'.format(params.holdout_fold))
SCORES_PATH = join(MODEL_DIR, 'scores_at_epoch_{}.csv'.format(args_checkpoint_epoch))
CONF_MAT_PATH = join(MODEL_DIR, 'cm_at_epoch_{}.csv'.format(args_checkpoint_epoch))

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rc('text', usetex=True)

# plotting scores
scores_df = pd.read_csv(SCORES_PATH)
LABELS = [s.replace('_', ' ') for s in scores_df['label']]
x = np.arange(params.classes_number)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

ax1.set_title('F1-scores per class for {} model using the validation dataset'.format(args_model_name),
        fontsize=18)
rects1 = ax1.bar(x, scores_df['F1_score'], width=0.8, edgecolor='black', alpha=0.7)
autolabel(rects1, ax1)
ax1.set_ylabel('F1-score', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, rotation=90)
ax1.set_ylim([0, 1.25])
ax1.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=14)

ax2.set_title('Accuracy scores per class for {} model using the validation dataset'.\
        format(args_model_name), fontsize=18)
rects2 = ax2.bar(x, scores_df['acc_score'], width=0.8, edgecolor='black', alpha=0.7)
autolabel(rects2, ax2)
ax2.set_ylabel('Accuracy score', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, rotation=90, fontsize=12)
ax2.set_ylim([0, 1.25])
ax2.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=14)

fig.tight_layout(pad=1.0)
fig.savefig(join(MODEL_DIR, 'scores_at_epoch_{}.pdf'.format(args_checkpoint_epoch)))

# plotting confusion matrix
conf_mat_df = pd.read_csv(CONF_MAT_PATH)
cm = conf_mat_df.values[:,1:]
cm_norm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]   # normalise

plot_confusion_matrix(cm=cm, normalise=False)
plot_confusion_matrix(cm=cm_norm, normalise=True)
