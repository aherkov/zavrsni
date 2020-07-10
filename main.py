from comet_ml import Experiment
import torch
from relay_net import ReLayNet
from solver import Solver
from data_loading import *

train_data, val_data, test_data = get_datasets()
print("Train size: %i" % len(train_data))
print("Val size: %i" % len(val_data))
print("Test size: %i" % len(test_data))

print(torch.cuda.is_available())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)

param = {
    'num_channels': 1,
    'num_filters': 64,
    'kernel_h': 5,
    'kernel_w': 3,
    'kernel_c': 1,
    'stride_conv': 1,
    'pool': 2,
    'stride_pool': 2,
    'num_class': 4
}

exp_dir_name = 'Exp01'

experiment = Experiment("9LRg8JI8I44sbVgkkIuUnTwXr", "ReLayNet")
relaynet_model = ReLayNet(param)
solver = Solver(optim_args={"lr": 0.1})
solver.train(relaynet_model, train_loader, val_loader, test_loader, exp_dir_name=exp_dir_name)
experiment.end()

relaynet_model.save("models/relaynet_model.model")
