# torch_template
A Python package for commonly used pytorch models, data loaders and utils. 


### Installation

Clone the repo, cd into it and run `pip install .` command.

``` bash
git clone https://github.com/misads/torch_template.git
cd torch_template
pip install . 
```

### Usage

```python
import torch_template
from torch_template.utils import misc_utils as utils
from torch_template.utils import torch_utils
```

### File structure

```yaml
File structure
    .
    ├── network
    │     ├── base_model.py     :Define models, losses and parameter updating
    │     ├── norm.py           :Normalizations
    │     └── weights_init.py   :weights init
    │
    ├── dataloader/             :Define Dataloaders
    ├── model_zoo               :Commonly used models
    └── utils
          ├── misc_utils.py     :System utils
          └── torch_utils.py    :PyTorch utils

```

