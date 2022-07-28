from diff_scm.configs import mnist_configs, brats_configs

def file_from_dataset(dataset_name):
    if dataset_name == "mnist":
        return mnist_configs.get_default_configs()
    elif dataset_name == "brats":
        return brats_configs.get_default_configs()
    else:
        raise Exception("Dataset not defined.")