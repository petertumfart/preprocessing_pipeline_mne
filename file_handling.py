import os


def gen_paths(pth, src_fldr, dst_fldr):
    """
    Generate source and destination paths for a file handling operation.

    :param pth: The path to the parent directory containing both the source and destination folders.
    :param src_fldr: The name of the folder containing the source files.
    :param dst_fldr: The name of the folder where output files will be saved.
    :return: A tuple containing the source path and destination path.
    """

    src_path = pth + src_fldr
    dst_path = pth + dst_fldr

    # Create dst folder if it does not exist:
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    return src_path, dst_path