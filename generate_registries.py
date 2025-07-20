# generate_registries.py
import os
import importlib.util

def scan_modules(directory, package_name):
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.{module_name}",
                os.path.join(directory, filename))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

if __name__ == "__main__":
    scan_modules("models", "models")
    scan_modules("metrics", "metrics")
    print("Registries updated successfully")