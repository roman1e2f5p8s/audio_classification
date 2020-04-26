from os.path import join
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

from hparams import HParamsFromYAML

args_validate = True
args_model_name = 'VGGish'

params = HParamsFromYAML('hparams.yaml', param_set=args_model_name)

if args_validate:
    MODEL_DIR = join(params.storage_dir, 'models', args_model_name, 'train_validation',
        'holdout_fold={}'.format(params.holdout_fold))
else:
    MODEL_DIR = join(params.storage_dir, 'models', args_model_name, 'train_only')

raw_df = pd.read_json(join(MODEL_DIR, 'data.json'), orient='row')

data_df = pd.DataFrame()
data_df['epoch'] = raw_df['train']['epoch']

for name, data in raw_df['train'][1:].iteritems():
    data_df['{}_t'.format(name)] = data

for name, data in raw_df['validation'][1:].iteritems():
    data_df['{}_v'.format(name)] = data


plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rc('text', usetex=True)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 9))

ax1.set_title('Loss over training for {} model'.format(args_model_name), fontsize=18)
ax1.plot(data_df['epoch'], data_df['loss_t'], '-o', color='blue', label='Training loss')
ax1.plot(data_df['epoch'], data_df['loss_v'], '-d', color='orange', label='Validation loss')
ax1.set_xticks(data_df['epoch'])
ax1.set_xlabel('epoch', fontsize=16)
ax1.set_ylabel('loss', fontsize=16)
ax1.legend()

ax2.set_title('Accuracy over training for {} model'.format(args_model_name), fontsize=18)
ax2.plot(data_df['epoch'], data_df['accuracy_t'], '-o', color='green', label='Training accuracy')
ax2.plot(data_df['epoch'], data_df['accuracy_v'], '-d', color='red', label='Validation accuracy')
ax2.set_xticks(data_df['epoch'])
ax2.set_xlabel('epoch', fontsize=16)
ax2.set_ylabel('accuracy', fontsize=16)
ax2.legend()

ax3.set_title('Weighted F1-score over training for {} model'.format(args_model_name), fontsize=18)
ax3.plot(data_df['epoch'], data_df['f1_score_t'], '-o', color='black', label='Training F1-score')
ax3.plot(data_df['epoch'], data_df['f1_score_v'], '-d', color='magenta', label='Validation F1-score')
ax3.set_xticks(data_df['epoch'])
ax3.set_xlabel('epoch', fontsize=16)
ax3.set_ylabel('F1-score', fontsize=16)
ax3.legend()

fig.tight_layout(pad=1.0)
fig.savefig(join(MODEL_DIR, 'data.pdf'))
