#!/usr/bin/env python
import os
import sys
import yaml
import argparse
import logging

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Hack to allow showing default values in help and preserving line break in epilog"""
    pass

parser = argparse.ArgumentParser(description='Execute a scheduler action locally given a config and a cycle',
                                 formatter_class=CustomFormatter)

parser.add_argument('-a', '--action', type=str, default='/ops/scheduler/actions/products/prod.storm_surge.yml',
                   help='Scheduler action yml file path. Ex: [/ops/scheduler/actions/products/prod.storm_surge.yml]')

parser.add_argument('-n', '--ncores', type=int, default=1,
                   help='Number of cores to be passet to set_mpiexec call')

parser.add_argument('-kw', '--kwargs', type=str, default=None,
                   help='Extra arguments to be updated and passed to the pycallable. e.g. kw1:val,kw2:val')

parser.add_argument('-l', '--loglevel', type=str, default='INFO',
                   help='The loglevel to run the action on')

def stdout_logger(loglevel=logging.DEBUG):
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = stdout_logger()

def import_pycallable(pycallable):
    pycallable = pycallable.split('.')
    method = pycallable[-1]
    module_str = '.'.join(pycallable[:-1])
    try:    
        module = il.import_module(module_str)
        return getattr(module, method)
    except ImportError:
        module = il.import_module(pycallable[0])
        nex_mod = getattr(module, pycallable[1])
        for submod in pycallable[2:]:
            nex_mod = getattr(nex_mod, submod)
        return nex_mod

def run_action(actionfile, cycle, ncores=1, loglevel='INFO', kwargs=None):
    if not os.path.exists(actionfile):
        print('Error: Action file does not exists')
        return sys.exit(1)

    logger.setLevel(loglevel)  
    config = yaml.load(open(actionfile))
    schedule = config.pop('schedule', None)
    pycallable = import_pycallable(config.pop('pycallable'))
    if kwargs:
        parsed = dict([map(yaml.load,kv.split(':')) for kv in kwargs.split(',')])
        config.update(parsed)
    instance = pycallable(logger=logger, **config)
    if hasattr(instance, 'set_mpiexec'):
        instance.set_mpiexec(ncores)
    if hasattr(instance, 'run'):
        result = instance.run()
    else:
        result = instance
    print('Action result: %s %s' % (os.linesep, str(result)))
    return result

def main():
    args = parser.parse_args()
    run_action(args.action, args.ncores, args.loglevel, args.kwargs)

if __name__=="__main__":
    main()