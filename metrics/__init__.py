# import glob
# import importlib
# import os

# modules = glob.glob(os.path.dirname(__file__)+'/*.py')
# for module in modules:
#     if not module.endswith('__init__.py'):
#         importlib.import_module(f'metrics.{os.path.basename(module)[:-3]}')