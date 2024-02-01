import importlib.metadata

class DistributionNotFound(Exception):
    pass

def require(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        raise DistributionNotFound(f"The '{package_name}' distribution was not found")

def get_distribution(package_name):
    try:
        return importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        raise DistributionNotFound(f"The '{package_name}' distribution was not found")