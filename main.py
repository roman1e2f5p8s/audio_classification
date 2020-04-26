import os
import json
import logging
from importlib import import_module
import numpy as np
import pandas as pd
from time import time
from shutil import copy2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.main_parser import Parser
from src.hparams import HParamsFromYAML
from src.logger import Logger
from src.models import VGGish
from src.data_generators import DataGenerator


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


def save_model(epoch, iteration, model, optimiser, path):   # TODO - put to tools
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


def send_evaluation_data(model, generator, device):  # TODO - put to utils
    '''
    Sends evaluation data to the model
    Parameters:
        - model -- model, object of a model class
        - generator -- generator of validation data, generator object
        - device -- to which device send the tensor, torch.device
    Returns:
        - dictionary {'filename': ..., 'target': ..., 'output': ...}, dict
    ''' 
    data = {'filename': [], 'target': [], 'output': [], 'losses': []}
    cost = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for (x_batch, y_batch, y, filename) in generator:
            data['filename'] += [filename]
            data['target'] += [y]
            x = array_to_tensor(x_batch, device)
            target = array_to_tensor(y_batch, device)
            model.eval()
            output = model(x)
            data['losses'] += cost(output, target)
            data['output'] += [output.data.cpu().numpy()]

    return {key: np.array(val) for key, val in data.items()}


def send_test_data(model, generator, device):  # TODO - put to utils
    '''
    Sends test data to the model
    Parameters:
        - model -- model, object of the VGGish class
        - generator -- generator of validation data, generator object
        - device -- to which device send the tensor, torch.device
    Returns:
        - dictionary {'filename': ..., 'output': ...}, dict
    ''' 
    data = {'filename': [], 'output': []}

    for (x_batch, filename) in generator:
        data['filename'] += [filename]
        x = array_to_tensor(x_batch, device)
        model.eval()
        output = model(x)
        data['output'] += [output.data.cpu().numpy()]   # TODO - why need cpu() here?

    return {key: np.array(val) for key, val in data.items()}


def mean_output(output):  # TODO - put to utils
    '''
    Returns the mean output over the number of chunks 
    Parameters:
        - output -- array of shape (audios_number,), numpy.ndarray
    Returns:
        - m_output -- array of shape (audios_number, classes_number), numpy.ndarray
    '''
    m_output = [np.mean(x, axis=0) for x in output]
    return np.array(m_output)


def evaluate(model, data_generator, validate, device, manually_verified_only, shuffle):
    '''
    Parameters:
        - model -- model, object of the VGGish class
        - data_generator -- data generator, object of the DataGenerator class
        - validate -- if true, use the validation dataset, else use params.eval_audios_number
            training data to evaluate the model, bool
        - device -- to which device send the tensor, torch.device
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the validation data, bool
    Returns:
        - (average_accuracy, F1_score, loss) -- accuracy classification score, F1-score and loss, tuple
    ''' 
    generator = data_generator.validation_generator(validate, manually_verified_only, shuffle)

    if validate:
        logging.info('Evaluation of the  model on validation data...')
    else:
        logging.info('Evaluation of the model on training data...')
    start_time = time()
    data = send_evaluation_data(model, generator, device)
    logging.info('Evaluation completed successfully. Elapsed time: {:.6f}'.format(time() - start_time))

    output = data['output']
    target = data['target']
    m_output = mean_output(output)
    predicted = np.argmax(m_output, axis=-1)
    loss = float(np.mean(data['losses']))

    average_accuracy = accuracy_score(target, predicted)
    F1_score = f1_score(target, predicted, average='weighted')

    return average_accuracy, F1_score, loss



def train(params, device, validate, manually_verified_only, shuffle, verbose):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - device -- to which device send the tensor, torch.device
        - validate -- whether split the dataset into train and validation sets, bool
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the validation data, bool
        - verbose -- if True, print the output to console, bool
    '''
    # path to train.h5 file with features
    HDF5_PATH = os.path.join(params.storage_dir, 'features', params.features_type, 'train.h5')
    
    # path where model will be saved
    if validate:
        MODEL_DIR = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type,
                'train_validation', 'holdout_fold={}'.format(params.holdout_fold))
    else:
        MODEL_DIR = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type,
                'train_only')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model = VGGish(params)
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
        logging.info('Iteration: {:5d}, epoch: {:3d}'.format(iteration, epoch))

        # evaluate model
        if iteration % data_generator.epoch_len == 0:
            if verbose:
                print('|****************************************************************************')
            eval_start_time = time()
            (acc, f1, loss) = evaluate(
                    model=model,
                    data_generator=data_generator,
                    validate=False,
                    device=device,
                    manually_verified_only=manually_verified_only,
                    shuffle=shuffle)
            logging.info('Training: accuracy={:.2f}, F1-score={:.2f}, loss={:.4f}'.format(acc, f1, loss))
            data['train']['epoch'] += [epoch]
            data['train']['accuracy'] += [acc]
            data['train']['f1_score'] += [f1]
            data['train']['loss'] += [loss]

            if validate:
                (acc, f1, loss) = evaluate(
                        model=model,
                        data_generator=data_generator,
                        validate=True,
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
                print('|****************************************************************************')
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

        # loss = F.nll_loss(output, y_batch)    # the negative log likelihood loss
        loss = F.cross_entropy(output, y_batch)

        logging.info('Backward propagation...')
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        iteration += 1
        # stop the generator and stop training if the limit of epochs has been reached
        if epoch > params.epochs_limit:
            break
        if verbose:
            print('|----------------------------------------------------------------------------')
    # save data
    with open(os.path.join(MODEL_DIR, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)
    # save hyperparameters
    copy2('hparams.yaml', os.path.join(MODEL_DIR, 'hparams.yaml'))


def get_class_scores(params, predicted, target):
    '''
    Calculates the correct and total number of labels, accuracy and F1-score for each class
    Parameters:
        - predicted -- array of predicted labels, numpy.ndarray
        - target -- array of target labels, numpy.ndarray
    Returns:
        - (LABELS, correct, total, accuracy, F1_score), tuple
    '''
    LABELS_JSON_PATH = os.path.join(params.storage_dir, params.labels_file)
    with open(LABELS_JSON_PATH, 'r') as f:
        LABELS = json.load(f)['labels']
    classes_number = params.classes_number

    correct= np.zeros(classes_number, dtype=np.int32)
    total = np.zeros(classes_number, dtype=np.int32)

    for i in range(len(target)):
        total[target[i]] += 1
        if predicted[i] == target[i]:
            correct[target[i]] += 1
    accuracy = correct.astype(np.float32) / total
    F1_score = f1_score(target, predicted, average=None)

    # TODO - remove logging
    logging.info('{:<25}{}/{}\t{}\t{}'.format('event labels', 'correct', 'total', 'acc', 'F1-score'))
    for (i, label) in enumerate(LABELS):
        logging.info('{:<25}{}/{}\t\t{:.2f}\t{:.2f}'.format(label, correct[i], total[i],
            accuracy[i], F1_score[i]))
    
    return LABELS, correct, total, accuracy, F1_score


def validate(params, device, checkpoint_epoch, manually_verified_only, shuffle):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - device -- to which device send the tensor, torch.device
        - checkpoint_epoch -- epoch from which to use saved model, int
        - manually_verified_only -- if True, use only manually verified audios for evaluation, bool
        - shuffle -- shuffle or not the validation data, bool
    Returns:
        - None
    '''
    SAVED_MODEL_PATH = os.path.join(params.storage_dir, 'models', params.model_name,
            params.features_type, 'train_validation', 'holdout_fold={}'.format(params.holdout_fold),
            'model_at_epoch_{}.pth'.format(checkpoint_epoch))
    HDF5_PATH = os.path.join(params.storage_dir, params.features_dir, params.features_type, 'train.h5')
    SCORES_PATH = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type,
            'train_validation', 'holdout_fold={}'.format(params.holdout_fold),
            'scores_at_epoch_{}.csv'.format(checkpoint_epoch))
    CONF_MAT_PATH = os.path.join(params.storage_dir, 'models', params.model_name, params.features_type,
            'train_validation', 'holdout_fold={}'.format(params.holdout_fold),
            'cm_at_epoch_{}.csv'.format(checkpoint_epoch))
    
    model = VGGish(params)
    saved_model = torch.load(SAVED_MODEL_PATH)
    model.load_state_dict(saved_model['model_state_dict'])
    model.to(device)

    data_generator = DataGenerator(params, validate=True)
    generator = data_generator.validation_generator(
            validate=True,
            manually_verified_only=manually_verified_only,
            shuffle=shuffle
            )

    logging.info('Sending the validation data to the model...')
    start_time = time()
    data = send_evaluation_data(model, generator, device)
    logging.info('Sending completed successfully. Elapsed time: {:.6f}'.format(time() - start_time))

    output = data['output']
    target = data['target']
    m_output = mean_output(output)
    predicted = np.argmax(m_output, axis=-1)
    loss = np.mean(data['losses'])

    average_accuracy = accuracy_score(target, predicted)
    F1_score = f1_score(target, predicted, average='weighted')
    conf_mat = confusion_matrix(target, predicted)

    logging.info('Iteration {}: accuracy score={:.2f}, F1-score={:.2f}, loss={:.4f}'.\
            format(checkpoint_epoch, average_accuracy, F1_score, loss))

    (LABELS, correct, total, accuracy, F1_score) = get_class_scores(params, predicted, target)

    result = pd.DataFrame()
    result['label'] = LABELS
    result['correct'] = correct
    result['total'] = total
    result['acc_score'] = accuracy
    result['F1_score'] = F1_score
    result.to_csv(SCORES_PATH)

    cm = pd.DataFrame(conf_mat)
    cm.to_csv(CONF_MAT_PATH)

    logging.info('Validation result saved to\n|{} and\n|{}'.format(SCORES_PATH, CONF_MAT_PATH))


def test():
    pass


def main():
    '''
    Main function
    Arguments:
        - None
    Returns:
        - None
    '''
    # create parser of arguments
    args = Parser().args

    # get hyperparameters
    params = HParamsFromYAML('hparams.yaml', param_sets=[args.features, args.model])

    logs_dir = os.path.join(params.logs_dir, args.model, args.features) 
    logger = Logger(logs_dir, verbose=args.verbose).logger

    # extract labels if labels.json does not exist yet
    if not os.path.isfile(os.path.join(params.storage_dir, params.labels_file)):
        from src.extract_labels import extract_labels
        extract_labels(params)

    # generate validation metadata if validation_meta.csv does not exist yet
    if args.mode == 'train' and args.validate:
        if not os.path.isfile(os.path.join(params.storage_dir, params.validation_dir,
                params.validation_meta_file)):
            from src.validation import generate_validation_metadata
            generate_validation_metadata(params)
    if args.mode == 'validation':
        if not os.path.isfile(os.path.join(params.storage_dir, params.validation_dir,
                params.validation_meta_file)):
            from src.validation import generate_validation_metadata
            generate_validation_metadata(params)

    if args.mode == 'train' or args.mode == 'test':
        # extract features if the corresponding h5 file does not exist yet
        if not os.path.isfile(os.path.join(params.storage_dir, params.features_dir, args.features,
                '{}.h5'.format(args.mode))):
            features_module = import_module('src.{}_features'.format(args.features))
            features_module.extract_features(args.mode)

    # for reproducibility
    torch.manual_seed(params.seed)

    # check whether cuda is available
    if args.cuda:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        if not cuda:
            logger.info('CUDA is not available. CPU will be used instead')
    else:
        device = torch.device('cpu')
        logger.info('Model will be sent to CPU')
    if args.verbose:
        print('|----------------------------------------------------------------------------')
    
    START_TIME = time()
    if args.mode == 'train':
        print('train')
        '''
        train(
                params=params,
                device=device,
                validate=args.validate,
                manually_verified_only=args.manually_verified_only,
                shuffle=args.shuffle,
                verbose=args.verbose
                )
        '''
    elif args.mode == 'validation':
        print('validation')
        '''
        validate(
                params=params,
                device=device,
                checkpoint_epoch=args.epoch,
                manually_verified_only=args.manually_verified_only,
                shuffle=args.shuffle
                )
        '''
    elif args.mode == 'test':
        print('test')
        # test()
    logger.info('Total time: {:.6f} s'.format(time() - START_TIME))
    if args.verbose:
        print('|----------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
