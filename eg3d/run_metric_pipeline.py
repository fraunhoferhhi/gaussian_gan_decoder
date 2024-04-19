import subprocess
import os
import argparse


class RunMetric:
    def __init__(self, args=[]):
        self.args = args
        self.python = "python"
        self.path_to_program = "calc_metrics.py"

    def run(self):
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    config_list = [
        RunMetric(["--network=/home/barthel/Documents/final_runs/eg3d/lpff/decoder_075000.pkl", "--use_decoder=True", "--mirror=False", "--camera_sample_mode=FFHQ_LPFF", "--data=/home/barthel/Documents/CVG3DGaussianHeads/code/model/eg3d/eg3d/datasets/FFHQ_LPFF"]),
        RunMetric(["--/home/barthel/Documents/final_runs/eg3d/ffhq/decoder_200000.pkl", "--use_decoder=True",
                   "--mirror=False", "--camera_sample_mode=FFHQ",
                   "--data=/home/barthel/Documents/CVG3DGaussianHeads/code/model/eg3d/eg3d/datasets/FFHQ_LPFF"]),

    ]

    for i in range(0, len(config_list), 1):
        config = config_list[i]
        print(f"Running with {config.args}")
        config.run()
