import inspect
import pkgutil

from badgers import generators
from badgers.core.base import GeneratorMixin



def is_implemented_method(method):
    """Check if a method is implemented and not abstract."""
    return not inspect.isabstract(method)


def list_defined_classes_and_methods(module):
    """
    List all classes and their methods defined in a given module.

    Args:
        module (module): The module to inspect.

    Returns:
        dict: A dictionary where keys are class names and values are tuples.
              Each tuple contains the class docstring and a list of method tuples.
              Each method tuple contains a method name and its docstring.
    """
    classes = {}

    # Walk through all submodules in the given module
    for loader, module_name, is_pkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        # Import the submodule
        mod = __import__(module_name, fromlist="dummy")

        # Inspect the submodule for classes and methods
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__module__ == module_name:  # Check if the class is defined in the module
                # Skip classes that directly inherit from GeneratorMixin
                if GeneratorMixin in obj.__bases__:
                    continue

                methods = []
                for m_name, m_obj in inspect.getmembers(obj, inspect.isfunction):
                    if (m_name == '__init__' or not m_name.startswith('_')) and is_implemented_method(
                        m_obj):  # Check if the method is public or __init__ and implemented
                        methods.append((m_name, m_obj.__doc__))  # Add the method name and docstring to the list
                if methods:  # Add only classes with implemented methods
                    classes[name] = (obj.__doc__, methods)  # Store class docstring and methods

    return classes


if __name__ == "__main__":
    # Call the function to get classes and methods from the badgers.generators module
    classes_and_methods = list_defined_classes_and_methods(generators)

    # Print the classes and their methods with docstrings
    for class_name, (class_doc, methods) in classes_and_methods.items():
        print(f"Class: {class_name}")
        if class_doc:
            print(f"  Class Docstring: {class_doc}")
        else:
            print(f"  Class Docstring: None")
        for method_name, method_doc in methods:
            print(f"  Method: {method_name}")
            if method_doc:
                print(f"    Docstring: {method_doc}")
            else:
                print(f"    Docstring: None")
