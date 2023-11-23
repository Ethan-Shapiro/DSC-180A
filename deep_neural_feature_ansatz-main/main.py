import numpy as np
import torch
import random
import dataset
import trainer

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


def get_name(dataset_name, configs):
    name_str = dataset_name + ':'
    for key in configs:
        name_str += key + ':' + str(configs[key]) + ':'
    name_str += 'nn'

    return name_str


def main():

    # Pick configs to save model
    configs = {}
    configs['num_epochs'] = 500
    configs['learning_rate'] = .1
    configs['weight_decay'] = 0
    configs['init'] = 'default'
    configs['optimizer'] = 'sgd'
    configs['freeze'] = False
    configs['width'] = 1024
    configs['depth'] = 2
    configs['act'] = 'relu'

    # Code to load and train net on selected dataset.
    # Datasets used in paper are in dataset.py
    # SVHN
    NUM_CLASSES = 2
    trainloader, valloader, testloader = dataset.get_celeba(
        feature_idx=16, num_train=12_000, num_test=1_000)
    accuracies = trainer.train_network(trainloader, valloader, testloader, NUM_CLASSES,
                                       name=get_name('celeba', configs), configs=configs)
    print(accuracies)


if __name__ == "__main__":
    main()
