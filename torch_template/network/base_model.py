import os
import torch
import sys

from misc_utils import color_print


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def set_input(self, input):
        self.input = input

    def forward(self, x):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            color_print("Exception: Checkpoint '%s' not found" % save_path, 1)
            if network_label == 'G':
                raise Exception("Generator must exist!,file '%s' not found" % save_path)
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
                color_print('Load checkpoint from %s.' % save_path, 3)
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate(self):
        pass
