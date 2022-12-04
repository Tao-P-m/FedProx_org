import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf

from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = [
    'fedavg',
    'fedprox',
    'feddane',
    'fedddane',
    'fedsgd',
    'fedprox_origin']
DATASETS = [
    'sent140',
    'nist',
    'shakespeare',
    'mnist',
    'synthetic_iid',
    'synthetic_0_0',
    'synthetic_0.5_0.5',
    'synthetic_1_1']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,),  # num_classes
    'sent140.stacked_lstm': (25, 2, 100),  # seq_len, num_classes, num_hidden
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100),
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,),  # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256),  # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, )  # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()
    # learning parameters
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default="mclr")  # 'stacked_lstm.py'
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=1)  # -1
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--M',
                        help='number of clients trained per round;',
                        type=int,
                        default=int(1))  # -1
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)
    # 01/09/2022 # simulation parameters added
    parser.add_argument('--SNR',
                        help='noise variance / 0.1W in dB',
                        type=float,
                        default=90)
    parser.add_argument('--verbose',
                        help='whether output or not',
                        type=int,
                        default=0)
    parser.add_argument('--set',
                        help=r'=1 if concentrated devices + euqal dataset; =2 if two clusters + unequal dataset',
                        type=int,
                        default=2)
    # optimization successive convex approximation
    parser.add_argument('--nit',
                        type=int,
                        default=100,
                        help=r'I_max, # of maximum SCA loops')
    parser.add_argument('--Jmax',
                        type=int,
                        default=50,
                        help=r'# of maximum Gibbs Outer loops')
    parser.add_argument('--threshold',
                        type=float,
                        default=1e-2,
                        help=r'epsilon, SCA early stopping criteria')
    parser.add_argument('--tau',
                        type=float,
                        default=1,
                        help=r'\tau, the SCA regularization term')
    parser.add_argument('--MC_trial', type=int, default=50,
                        help=r'total number of Monte Carlo trials')
    parser.add_argument('--N', type=int, default=5, help='# of BS antennas')
    parser.add_argument('--L', type=int, default=40, help='RIS Size')

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.random.set_seed(123 + parsed['seed'])

    # load selected model
    if parsed['dataset'].startswith(
            "synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    parsed['alpha_direct'] = 3.76  # User-BS Path loss exponent
    # carrier frequency, wavelength lambda=3.0*10**8/fc
    parsed['fc'] = 915 * 10**6
    parsed['BS_Gain'] = 10**(5.0 / 10)  # BS antenna gain
    parsed['RIS_Gain'] = 10**(5.0 / 10)  # RIS antenna gain
    parsed['User_Gain'] = 10**(0.0 / 10)  # User antenna gain
    parsed['d_RIS'] = 1.0 / 10  # dimension length of RIS element/wavelength
    parsed['BS'] = np.array([-50, 0, 10])  # the PS is placed at
    parsed['RIS'] = np.array([0, 0, 10])  # the RIS is placed at

    parsed['range'] = 20
    parsed['transmitPower'] = 1.0 # 0.1

    x0 = np.ones([parsed['M']], dtype=int)  # for device selection

    parsed['SCA_Gibbs'] = np.ones(
        [parsed['Jmax'] + 1, parsed['MC_trial']]) * np.nan
    parsed['DC_NORIS_set'] = np.ones([parsed['MC_trial'], ]) * np.nan
    parsed['DC_NODS_set'] = np.ones([parsed['MC_trial'], ]) * np.nan
    parsed['Alt_Gibbs'] = np.ones(
        [parsed['Jmax'] + 1, parsed['MC_trial']]) * np.nan
    parsed['DG_NORIS'] = np.ones([parsed['MC_trial'], ]) * np.nan

    parsed['ref'] = (1e-10)**0.5
    sigma = np.power(10, -parsed['SNR'] / 10)
    parsed['sigma'] = sigma / parsed['ref']**2

    return parsed, learner, optimizer


def main():
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    # parse command line arguments
    options, learner, optimizer = read_options()  # parsed, learner, optimizer

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()


if __name__ == '__main__':
    main()
