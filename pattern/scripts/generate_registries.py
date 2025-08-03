# generate_registries.py
import os
import importlib.util
from pathlib import Path

base_path = Path(__file__).parent.parent / "config" / "registries"
searchfile = Path(__file__).parent.parent

def scan_modules(directory, package_name):
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.{module_name}",
                os.path.join(directory, filename))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

def main():
    scan_modules(f"{searchfile / 'models/'}", f"{base_path / 'models'}")
    scan_modules(f"{searchfile / 'metrics/'}",f"{base_path / 'metrics'}")
    print(f"Registries updated successfully, {base_path / 'models'} {base_path / 'metrics'}")
    
if __name__ == "__main__":
    main()
    