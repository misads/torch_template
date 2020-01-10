# torch_template

<p>
    <a href='https://torch-template.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/torch-template/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License">
    </a>
</p>

A python package for commonly used pytorch models, data loaders and utils. 


### Installation

**For pip**  

```bash
pip install torch-template
```

**For source**

Clone the repo, cd into it and run `pip install .` command.

```bash
git clone https://github.com/misads/torch_template.git
cd torch_template
pip install .
```

**For conda**

```bash
source ~/anaconda3/bin/activate
conda activate <env>
python setup.py install
```

A configure file `torch_template.egg-info` will be generated in the repo directory. Copy `torch_template` and `torch_template.egg-info` to your `site-packages` folder.

### Usage

Test if the package is successfully installed:

```python
import torch_template as tt
from torch_template import torch_utils
```

Run:

```bash
#!-bash
tt-new
```

Enter your repo name, then a python project template will be created.

### Documentation

The documentation webpage can be found here <https://torch-template.readthedocs.io/en/latest/>.

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
          └── torch_utils.py    :PyTorch utils
          
```

