#!env python3
import argparse
import os
import json
import sys
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Release roadstar', prog='release')
    parser.add_argument(
        '-c', '--config', metavar='config_file', help='config file path')
    parser.add_argument('-r', '--root', metavar='root', default='/roadstar',
                        help='root path of project (default: /roadstar)')
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print('project root {} not found'.format(args.root), file=sys.stderr)
        exit(-1)

    if not os.path.isabs(args.root):
        print('project root {} is not absolute'.format(
            args.root), file=sys.stderr)
        exit(-1)

    config = {}
    if args.config is not None and os.path.exists(args.config):
        with open(args.config) as file:
            config = json.load(file)

    os.chdir(args.root)
    if 'base' in config:
        result = os.system('git checkout {}'.format(config['base']))
        if result is not 0:
            exit(result)

    result = os.system('git submodule update')
    if result is not 0:
        exit(result)

    if 'submodules' in config:
        for submodule, commit in config['submodules'].items():
            submodule_path = os.path.join(args.root, submodule)
            if not os.path.exists(submodule_path):
                print('submodule path {} not found'.format(
                    submodule_path), file=sys.stderr)
                exit(-1)
            os.chdir(submodule_path)
            result = os.system('git checkout {}'.format(commit))
            if result is not 0:
                exit(result)

    os.chdir(args.root)

    result = os.system(
        'git checkout -b rc_{0} && git commit -am "release" && git push origin rc_{0}'.format(datetime.date.today()))
    if result is not 0:
        exit(result)

    pass
