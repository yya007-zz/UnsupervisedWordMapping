from cyclegan-text.CycleGAN import *

if __name__ == '__main__':
    to_train=1
    log_dir='./output/cyclegan/exp_01'
    config_filename='./configs/exp_01.json'
    checkpoint_dir='./output/cyclegan/exp_01/#timestamp#'
    run_cyclegan(to_train, log_dir, config_filename, checkpoint_dir, skip)