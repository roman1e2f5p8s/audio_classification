import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from time import time
from shutil import copy2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_memlab import MemReporter

from .data_generators import DataGenerator


def reproducibility(seed):
    '''
    Makes runs reproducible
    Parameters:
        - seed -- seed, int
    Returns:
        - None
    '''
    torch.manual_seed(seed)


def get_device(cuda_flag):
    '''
    Checks if cuda is available
    Parameters:
        - cuda_flag -- use cuda or not, bool
    Returns:
        - device -- device to which the tensors will be sent, torch.device
    '''
    if cuda_flag:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        if not cuda:
            logging.info('CUDA is not available. CPU will be used instead')
    else:
        device = torch.device('cpu')
        logging.info('Model will be sent to CPU')
    return device


def array_to_tensor(x, device):
    '''
    Converts a numpy array to a torch tensor
    Parameters:
        - x -- array, numpy.ndarray
        - device -- to which device send the tensor, torch.device
    Returns:
        - torch.FloatTensor or torch.LongTensor
    '''
    x_ = torch.FloatTensor(x) if 'float' in str(x.dtype) else torch.LongTensor(x)
    return x_.cuda() if device == torch.device('cuda') else x_


def count_model_params(model):
    '''
    Counts the number of trainable parameters
    Parameters:
        - model -- model, object of a model class
    Returns:
        - number of trainable parameters, int
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grab_memory_usage_output(model):
    '''
    Grabs the output of pytorch MemReporter
    Parameters:
        - model -- model, object of a model class
    Returns:
        - None
    '''
    reporter = MemReporter(model)

    orig_stdout = sys.stdout  # remebmer the original stdout

    with open('temp.txt', 'w') as f:
        sys.stdout = f
        reporter.report(verbose=True)
    with open('temp.txt', 'r') as f:
        lines = f.readlines()
    sys.stdout = orig_stdout  # switch to the original stdout
    os.remove('temp.txt')

    lines[-1] = lines[-1][:-1]
    new_lines = ['|{}'.format(line) for line in lines]
    logging.info('Memory consuming:\n{}'.format(''.join(line for line in new_lines)))


def save_model(epoch, iteration, model, optimiser, path):
    '''
    Save the model at given epoch and iteration to .pth file
    Parameters:
        - epoch -- at which epoch save the model
        - iteration -- at which iteration save the model
        - model -- model, object of a model class
        - optimiser -- optimiser, torch.optim.adam.Adam
        - path -- path to file where the model will be saved, srt
    '''
    data = {'epoch': epoch, 'iteration': iteration, 'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict()}
    torch.save(data, path)


def send_evaluation_data(model, generator, device):
    '''
    Sends evaluation data to the model
    Parameters:
        - model -- model, object of a model class
        - generator -- generator of validation data, generator object
        - device -- to which device send the tensor, torch.device
    Returns:
        - dictionary {'target': ..., 'output': ..., 'losses': ...}, dict
    ''' 
    data = {'target': [], 'output': [], 'losses': []}
    cost = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for (x_batch, y_batch, y) in generator:
            data['target'] += [y]
            x = array_to_tensor(x_batch, device)
            target = array_to_tensor(y_batch, device)
            model.eval()
            output = model(x)
            data['losses'] += [cost(output, target).data.cpu().numpy()]
            data['output'] += [output.data.cpu().numpy()]

    return {key: np.array(val) for key, val in data.items()}


def send_test_data(model, generator, device):
    '''
    Sends test data to the model
    Parameters:
        - model -- model, object of a model class
        - generator -- generator of validation data, generator object
        - device -- to which device send the tensor, torch.device
    Returns:
        - dictionary {'filename': ..., 'output': ..., 'target': ...}, dict
    ''' 
    data = {'filename': [], 'output': [], 'target': []}

    with torch.no_grad():
        for (x_batch, filename, label) in generator:
            data['target'] += [label]
            data['filename'] += [filename]
            x = array_to_tensor(x_batch, device)
            model.eval()
            output = model(x)
            data['output'] += [output.data.cpu().numpy()]

    return {key: np.array(val) for key, val in data.items()}


def mean_output(output):
    '''
    Returns the mean output over the number of chunks 
    Parameters:
        - output -- array of shape (audios_number,), numpy.ndarray
    Returns:
        - m_output -- array of shape (audios_number, classes_number), numpy.ndarray
    '''
    m_output = [np.mean(x, axis=0) for x in output]
    return np.array(m_output)


def evaluate(model, data_generator, mode, device, manually_verified_only, shuffle):
    '''
    Parameters:
        - model -- model, object of a model class
        - data_generator -- data generator, object of the DataGenerator class
        - mode -- 'train' or 'validation', str
        - device -- to which device send the tensor, torch.device
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the evaluation data, bool
    Returns:
        - (average_accuracy, F1_score, loss) -- accuracy classification score, F1-score and loss, tuple
    ''' 
    generator = data_generator.validation_generator(
            mode=mode,
            manually_verified_only=manually_verified_only,
            shuffle=shuffle
            )

    start_time = time()
    data = send_evaluation_data(model, generator, device)
    logging.info('Evaluation completed successfully.\n|Elapsed time: {:.6f}'.format(time() - start_time))

    output = data['output']
    target = data['target']
    m_output = mean_output(output)
    predicted = np.argmax(m_output, axis=-1)
    loss = float(np.mean(data['losses']))

    average_accuracy = accuracy_score(target, predicted)
    F1_score = f1_score(target, predicted, average='weighted')

    return average_accuracy, F1_score, loss


def train(params, model, device, validate, manually_verified_only, shuffle, verbose):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - model -- model, object of a model class
        - device -- to which device send the tensor, torch.device
        - validate -- whether split the dataset into train and validation sets, bool
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the evaluation data, bool
        - verbose -- if True, print the output to console, bool
    Returns:
        - None
    '''
    # get path where model will be saved
    BASE_PATH = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type)
    subbase_path = os.path.join('train_validation', 'holdout_fold={}'.format(params.holdout_fold)) \
            if validate else 'train_only'
    subbase_path = os.path.join(subbase_path, 'manually_verified_only') if manually_verified_only \
            else os.path.join(subbase_path, 'all')
    MODEL_DIR = os.path.join(BASE_PATH, subbase_path)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model.to(device)

    optimiser = optim.Adam(
            params=model.parameters(),
            lr=params.learning_rate,
            betas=(params.beta_1, params.beta_2),
            eps=params.eps,
            weight_decay=params.weight_decay,
            amsgrad=params.amsgrad
            )

    data_generator = DataGenerator(params, validate=validate)

    iteration = 1
    train_start_time = time()
    data = {'train': {'epoch': [], 'accuracy': [], 'f1_score': [], 'loss': []}}
    if validate:
        data['validation'] = {'epoch': [], 'accuracy': [], 'f1_score': [], 'loss': []}

    for (x_batch, y_batch) in data_generator.train_generator():
        epoch = data_generator.epoch
        logging.info('Iteration: {:5d}, epoch: {:3d}, epochs_limit: {:3d}'.format(iteration, epoch,
                params.epochs_limit))

        # evaluate and save model each epoch
        if iteration % data_generator.epoch_len == 0:
            if verbose:
                print('|*******************************************************************************')
            eval_start_time = time()
            logging.info('Evaluation of the model on training data...')
            (acc, f1, loss) = evaluate(
                    model=model,
                    data_generator=data_generator,
                    mode='train',
                    device=device,
                    manually_verified_only=manually_verified_only,
                    shuffle=shuffle)
            logging.info('Training: accuracy={:.3f}, F1-score={:.3f}, loss={:.4f}'.format(acc, f1, loss))
            data['train']['epoch'] += [epoch]
            data['train']['accuracy'] += [acc]
            data['train']['f1_score'] += [f1]
            data['train']['loss'] += [loss]

            if validate:
                logging.info('Evaluation of the  model on validation data...')
                (acc, f1, loss) = evaluate(
                        model=model,
                        data_generator=data_generator,
                        mode='validation',
                        device=device,
                        manually_verified_only=manually_verified_only,
                        shuffle=shuffle)
                logging.info('Validation: accuracy={:.3f}, F1-score={:.3f}, loss={:.4f}'.format(acc, f1,
                        loss))
                data['validation']['epoch'] += [epoch]
                data['validation']['accuracy'] += [acc]
                data['validation']['f1_score'] += [f1]
                data['validation']['loss'] += [loss]

            logging.info('Elapsed time: training={:.3f} s, evaluation={:.3f} s'.format(
                    eval_start_time - train_start_time, time() - eval_start_time))

            # save model:
            save_model_file_path = os.path.join(MODEL_DIR, 'model_at_epoch_{}.pth'.format(epoch))
            logging.info('Saving model...')
            save_model(
                    epoch=epoch,
                    iteration=iteration,
                    model=model,
                    optimiser=optimiser,
                    path=save_model_file_path)
            logging.info('Model saved successfully to\n|{}'.format(save_model_file_path))
            if verbose:
                print('|*******************************************************************************')
            train_start_time = time()

        # reduce learning rate
        if iteration % params.learning_rate_decay_step == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] *= params.learning_rate_decay

        x_batch = array_to_tensor(x_batch, device)
        y_batch = array_to_tensor(y_batch, device)

        logging.info('Forward propagation...')
        model.train()
        output = model(x_batch)

        loss = F.cross_entropy(output, y_batch)

        logging.info('Backward propagation...')
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if iteration == 1:
            # check how much memory the model consumes
            grab_memory_usage_output(model)

        iteration += 1
        # stop the generator and stop training if the limit of epochs has been reached
        if epoch > params.epochs_limit:
            break
        if verbose:
            print('|-------------------------------------------------------------------------------')

    # save data
    data_path = os.path.join(MODEL_DIR, 'data.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info('Accuracies and losses saved to\n|{}'.format(data_path))

    # save hyper-parameters
    hparams_path = os.path.join(MODEL_DIR, 'hparams.yaml')
    copy2('hparams.yaml', hparams_path)
    logging.info('Hyper-parameters saved to\n|{}'.format(hparams_path))
    if verbose:
        print('|-------------------------------------------------------------------------------')


def get_class_scores(params, predicted, target):
    '''
    Calculates the correct and total number of labels, accuracy and F1-score for each class
    Parameters:
        - predicted -- array of predicted labels, numpy.ndarray
        - target -- array of target labels, numpy.ndarray
    Returns:
        - (correct, total, accuracy, F1_score), tuple
    '''
    classes_number = params.classes_number

    correct= np.zeros(classes_number, dtype=np.int32)
    total = np.zeros(classes_number, dtype=np.int32)

    for i in range(len(target)):
        total[target[i]] += 1
        if predicted[i] == target[i]:
            correct[target[i]] += 1
    accuracy = correct.astype(np.float32) / total
    F1_score = f1_score(target, predicted, average=None)

    return correct, total, accuracy, F1_score


def validate(params, model, device, checkpoint_epoch, validated, manually_verified_only, shuffle,
        verbose):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - model -- model, object of a model class
        - device -- to which device send the tensor, torch.device
        - checkpoint_epoch -- epoch from which to use saved model, int
        - validated -- where the model was trained on full train dataset or not, bool
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the validation data, bool
        - verbose -- if True, print the output to console, bool
    Returns:
        - None
    '''
    BASE_PATH = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type)
    subbase_path = os.path.join('train_validation', 'holdout_fold={}'.format(params.holdout_fold)) \
            if validate else 'train_only'
    subbase_path = os.path.join(subbase_path, 'manually_verified_only') if manually_verified_only \
            else os.path.join(subbase_path, 'all')
    # to load the model
    SAVED_MODEL_PATH = os.path.join(BASE_PATH, subbase_path,
            'model_at_epoch_{}.pth'.format(checkpoint_epoch))
    # to save the results
    SCORES_PATH = os.path.join(BASE_PATH, subbase_path,
            'val_scores_at_epoch_{}.csv'.format(checkpoint_epoch))
    CONF_MAT_PATH = os.path.join(BASE_PATH, subbase_path,
            'val_cm_at_epoch_{}.csv'.format(checkpoint_epoch))
    
    saved_model = torch.load(SAVED_MODEL_PATH)
    model.load_state_dict(saved_model['model_state_dict'])
    model.to(device)

    data_generator = DataGenerator(params, validate=True, verbose=verbose)
    generator = data_generator.validation_generator(
            mode='validation',
            manually_verified_only=manually_verified_only,
            shuffle=shuffle
            )

    logging.info('Sending the validation data to the model...')
    start_time = time()
    data = send_evaluation_data(model, generator, device)
    logging.info('Sending completed successfully.\n|Elapsed time: {:.6f}'.format(time() - start_time))

    output = data['output']
    target = data['target']
    m_output = mean_output(output)
    predicted = np.argmax(m_output, axis=-1)
    loss = np.mean(data['losses'])

    average_accuracy = accuracy_score(target, predicted)
    F1_score = f1_score(target, predicted, average='weighted')
    conf_mat = confusion_matrix(target, predicted)

    logging.info('Checkpoint epoch {}: accuracy score={:.2f}, F1-score={:.2f},\n|loss={:.4f}'.\
            format(checkpoint_epoch, average_accuracy, F1_score, loss))

    (correct, total, accuracy, F1_score) = get_class_scores(params, predicted, target)

    # save scores
    result = pd.DataFrame()
    LABELS_JSON_PATH = os.path.join(params.storage_dir, params.labels_file)
    with open(LABELS_JSON_PATH, 'r') as f:
        result['label'] = json.load(f)['labels']
    result['correct'] = correct
    result['total'] = total
    result['acc_score'] = accuracy
    result['F1_score'] = F1_score
    result.to_csv(SCORES_PATH)

    # save confusion matrix
    cm = pd.DataFrame(conf_mat)
    cm.to_csv(CONF_MAT_PATH)

    logging.info('Validation results saved to\n|{} and\n|{}'.format(SCORES_PATH, CONF_MAT_PATH))


def test(params, model, device, checkpoint_epoch, validated, manually_verified_only, verbose):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - model -- model, object of a model class
        - device -- to which device send the tensor, torch.device
        - checkpoint_epoch -- epoch from which to use saved model, int
        - validated -- where the model was trained on full train dataset or not, bool
        - manually_verified_only -- test model if it was trained using only manually verified audios
            for evaluation, bool
        - verbose -- if True, print the output to console, bool
    Returns:
        - None
    '''
    BASE_PATH = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type)
    subbase_path = os.path.join('train_validation', 'holdout_fold={}'.format(params.holdout_fold)) \
            if validated else 'train_only'
    subbase_path = os.path.join(subbase_path, 'manually_verified_only') if manually_verified_only \
            else os.path.join(subbase_path, 'all')
    # to load the model
    SAVED_MODEL_PATH = os.path.join(BASE_PATH, subbase_path,
            'model_at_epoch_{}.pth'.format(checkpoint_epoch))
    # to save the results
    SCORES_PATH = os.path.join(BASE_PATH, subbase_path,
            'test_scores_at_epoch_{}.csv'.format(checkpoint_epoch))
    CONF_MAT_PATH = os.path.join(BASE_PATH, subbase_path,
            'test_cm_at_epoch_{}.csv'.format(checkpoint_epoch))
    TEST_SUBM_PATH = os.path.join(BASE_PATH, subbase_path,
            'test_subm_at_epoch_{}.csv'.format(checkpoint_epoch))

    saved_model = torch.load(SAVED_MODEL_PATH)
    model.load_state_dict(saved_model['model_state_dict'])
    model.to(device)

    data_generator = DataGenerator(params, validate=False, verbose=verbose)
    data_generator.read_test_data()
    generator = data_generator.test_generator()

    logging.info('Sending the test data to the model...')
    start_time = time()
    data = send_test_data(model, generator, device)
    logging.info('Sending completed successfully.\n|Elapsed time: {:.6f}'.format(time() - start_time))

    filenames = data['filename']
    output = data['output']
    target = data['target']
    m_output = mean_output(output)
    predicted = np.argmax(m_output, axis=-1)

    average_accuracy = accuracy_score(target, predicted)
    F1_score = f1_score(target, predicted, average='weighted')
    conf_mat = confusion_matrix(target, predicted)

    logging.info('Checkpoint epoch {}: accuracy score={:.2f}, F1-score={:.2f}'.\
            format(checkpoint_epoch, average_accuracy, F1_score))

    (correct, total, accuracy, F1_score) = get_class_scores(params, predicted, target)

    LABELS_JSON_PATH = os.path.join(params.storage_dir, params.labels_file)
    with open(LABELS_JSON_PATH, 'r') as f:
        LABELS = json.load(f)['labels']

    # save scores
    result = pd.DataFrame()
    result['label'] = LABELS 
    result['correct'] = correct
    result['total'] = total
    result['acc_score'] = accuracy
    result['F1_score'] = F1_score
    result.to_csv(SCORES_PATH)

    # save confusion matrix
    cm = pd.DataFrame(conf_mat)
    cm.to_csv(CONF_MAT_PATH)

    # save test submission
    subm = pd.DataFrame()
    subm['filename'] = filenames
    subm['label'] = [LABELS[target[n]] for n in range(len(filenames))]
    subm['predicted'] = [LABELS[predicted[n]] for n in range(len(filenames))]
    subm.to_csv(TEST_SUBM_PATH)

    logging.info('Test result saved to\n|{},\n|{}, and\n|{}'.format(SCORES_PATH, CONF_MAT_PATH,
            TEST_SUBM_PATH))
