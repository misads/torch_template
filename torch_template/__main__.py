import argparse
import os
import shutil

from pkg_resources import resource_filename
from utils import misc_utils as utils


def replace(dst, name):
    flag = False
    with open(dst, 'r') as f:
        text = f.read()
        if '{{name}}' in text:
            flag = True
        text = text.replace('{{name}}', name)

    if flag:
        with open(dst, 'w') as f:
            f.write(text)


def init_repo(path):
    print("### torch_template project initializer ###")

    if path is None:
        name = input("Please enter your repo name: ")
        path = os.path.join(os.curdir, name)
        if os.path.exists(path):
            path = os.path.abspath(path)
            utils.color_print("The path '%s' already exists, aborting!" % path, 1)
            return
    else:
        name = utils.get_file_name(path)
    path = os.path.abspath(path)

    print("\nCreating Project '%s' into '%s'\n" % (name, path))

    subfolders = {"dataloader": ['dataset.py'],
                  "datasets": [],
                  "network": ['Model.py'],
                  'options': ['__init__.py', 'options.py']}

    print("Setting up folder structure:")
    utils.color_print(" %s" % os.path.basename(path), 7)
    os.makedirs(path, exist_ok=True)
    prefix = ' ' * (len(name) // 2)

    promts = []

    ##########################
    #        Subfolders
    ##########################
    for idx, s in enumerate(subfolders):
        print(prefix + " ├── ", end='')
        utils.color_print(s, 7)
        os.makedirs(os.path.join(path, s), exist_ok=True)
        if any(subfolders[s]):
            for idx_sub, s_sub in enumerate(subfolders[s]):
                if idx_sub == len(subfolders[s]) - 1:
                    print(prefix + " │      └── ", end='')
                    utils.color_print(s_sub, 2)
                else:
                    print(prefix + " │      ├── ", end='')
                    utils.color_print(s_sub, 2)
                fname = resource_filename(__name__, os.path.join('templates', s, s_sub))
                dst = os.path.join(path, s, s_sub)
                shutil.copy(fname, dst)
                promts.append("Copy '%s' to '%s'" % (os.path.join('templates', s, s_sub), dst))

    ############################
    #   Files in repo root dir
    ############################
    files = ['train.py', 'eval.py', 'test.py', 'clear.py', 'README.md']

    for idx, s in enumerate(files):
        if idx == len(files) - 1:
            print(prefix + " └── ", end='')
            utils.color_print(s, 2)
        else:
            print(prefix + " ├── ", end='')
            utils.color_print(s, 2)
        fname = resource_filename(__name__, os.path.join('templates', s))
        dst = os.path.join(path, s)
        shutil.copy(fname, dst)
        promts.append("Copy '%s' to '%s'" % (os.path.join('templates', s), dst))
        replace(dst, name)

    print()
    for i in promts:
        print(i)
    print('Finished.')

    exit(1)
    env = Environment(
        loader=PackageLoader('torch_template', 'templates'),
    )

    print("Initializing repo files")
    for f in ["README.md"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["models.py", "datasets.py", "interfaces.py", "callbacks.py",
              "version.py", "__init__.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, name, f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["train.py", "eval.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, "scripts", f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["test_basic.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, "tests", f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    fname = resource_filename('torch_template', os.path.join('templates', "default.yml"))
    dst = os.path.join(path, "config", "default.yml")
    shutil.copy(fname, dst)

    print("Done! Check the readme at %s" % os.path.join(path, "README.md"))


def new_project():
    parser = argparse.ArgumentParser(
        description="initialized a directory for a new project.")
    parser.add_argument("--path", help="path to the root of the repo.")
    args = parser.parse_args()

    init_repo(args.path)


if __name__ == '__main__':
    new_project()
