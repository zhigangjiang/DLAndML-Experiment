import os


def get_best_checkpoint_path(checkpoint_dir):

    best_mode_path = os.path.join(checkpoint_dir, sorted([file_name for file_name in os.listdir(checkpoint_dir)
                                                          if file_name.__contains__(".pth")],
                                                         key=lambda file_name: float(
                                                             file_name.split("_")[-1].split(".pth")[0]),
                                                         reverse=True)[0])
    return best_mode_path
