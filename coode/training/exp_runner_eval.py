import sys
sys.path.append('../coode')
import argparse
import GPUtil

from training.pir_eval import PIRTestRunner
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=1001, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='/media/disk2/smq_data/TransPIR/coode/confs/fixed_cameras_eval.conf')

    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=True, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='cat-eval', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='1000',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        # gpu = deviceIDs[0]
        gpu = 2
    else:
        gpu = opt.gpu
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致


    trainrunner = PIRTestRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=gpu,
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 train_cameras=opt.train_cameras
                                 )

    trainrunner.run()