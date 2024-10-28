import torch.nn as nn


class ListModule(object):
    # Should work with all kind of module
    def __init__(self, module: nn.Module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self._num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module: nn.Module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        else:
            self.module.add_module(self.prefix + str(self._num_module), new_module)
            self._num_module += 1

    def __len__(self):
        return self._num_module

    def __getitem__(self, i: int):
        if i < 0 or i >= self._num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))
