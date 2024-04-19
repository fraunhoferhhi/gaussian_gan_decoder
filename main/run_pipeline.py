import subprocess
import os
import argparse


class TrainDecoder:
    def __init__(self, args=[]):
        self.args = args
        self.args.append("--use_wandb=true")
        self.python = "python"
        self.path_to_program = "train_pano2gaussian_decoder.py"

    def run(self):
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    gpu_index = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"


    group_name = "--group=hu_server_eg3d"
    config_list = [
        TrainDecoder([group_name, "--generator_arch=eg3d_ffhq", "--run_name=eg3d_ffhq_long", "--num_iter=400001", "--load_checkpoint=/home/tmp/barthefl/projects/FullGANDecoder/main/results/eg3d/runeg3d_ffhq_long_23/decoder_100000.pkl"]),
        TrainDecoder([group_name, "--generator_arch=eg3d_lpff", "--run_name=eg3d_lpff_long", "--num_iter=400001", "--load_checkpoint=/home/tmp/barthefl/projects/FullGANDecoder/main/results/eg3d/runeg3d_lpff_long_26/decoder_100000.pkl"]),
        TrainDecoder([group_name, "--generator_arch=eg3d_cats", "--run_name=eg3d_cats_long", "--num_iter=400001", "--load_checkpoint=/home/tmp/barthefl/projects/FullGANDecoder/main/results/eg3d/runeg3d_cats_long_25/decoder_100000.pkl"]),

    ]

    for i in range(0, len(config_list), 4):
        config = config_list[i + gpu_index]
        print(f"Running with {config.args}")
        config.run()
