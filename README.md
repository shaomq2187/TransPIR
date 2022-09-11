# TransPIR
This repository contains the implementation for our paper "Polarimetric Inverse Rendering for Transparent Shapes Reconstruction". In this work, we propose a method for the detailed reconstruction of transparent objects by exploiting polarimetric cues.  

[Paper](https://arxiv.org/pdf/2208.11836.pdf)|[Code](https://github.com/shaomq2187/TransPIR)[|Dataset](https://cloud.tsinghua.edu.cn/f/2feaea15e9094941b4bd/?dl=1)

<img src="figures/cat-shape.gif" alt="cat-shape" width=200 /><img src="figures/elephant-shape.gif" alt="elephant-shape" width=200 /><img src="figures/frog-shape.gif" alt="frog-shape" width=200 /><img src="figures/squirrel-shape.gif" alt="squirrel-shape" width=200 />

<img src="figures/cat-transparent.gif" alt="cat-transparent" width=200 /><img src="figures/elephant-transparent.gif" alt="elephant-transparent" width=200 /><img src="figures/frog-transparent.gif" alt="frog-transparent" width=200 /><img src="figures/squirrel-transparent.gif" alt="squirrel-transparent" width=200 />

## Installation and Requirements

### Environment

The environment required to run this code can be created by the following commands:

```
conda env create -f environment.yml
conda activate TransPIR
```

### Dataset

We  construct TransPIR dataset which consists of 4 transparent objects: `cat、frog、elephant and squirrel`. The TransPIR dataset can be download using:

```
sh download_data.sh
```

The structure and the meaning of the  subfolders are shown below:

<details> <summary>Details</summary> <pre><code>.data
├── cat
|	├── dummy # polarization image with angle of polarizer 0°, (1232x1028,UInt8)
│   │   ├── cameras.npz # numpy file that contains camera poses
|	├── I-0 # polarization image with angle of polarizer 0°, (1232x1028,UInt8)
|	├── I-45 # polarization image with angle of polarizer 45°, (1232x1028,UInt8)
|	├── I-90 # polarization image with angle of polarizer 90°, (1232x1028,UInt8)
|	├── I-135 # polarization image with angle of polarizer 135°, (1232x1028,UInt8)
│   ├── I-sum # intensity image, (1232x1028,UInt8)
│   ├── masks # foreground mask  (1232x1028,UInt8)
│   ├── normals-png # ground truth surface normals,[-1,1] to [0,255] (1232x1028x3,UInt8)
│   ├── params #polarization params
│   │   ├── AoLP # angle of linear polarization, [0°,180°] to [0,255] (1232x1028,UInt8)
│   │   ├── DoLP # degree of linear polarization, [0,1] to [0,255] (1232x1028,UInt8)
│   ├── cameras_new.npz # numpy file that contains normalized camera poses
├── elephant
├── frog
├── squirrel </code>
</pre> </details>

###  Pretrained Models

We provide the pretrained model of the four objects, which can be downloaded through the following command:

```
sh download_pretrained_models.sh
```

The downloaded pretrained models will store in the `./exps/fixed_cameras/cat-eval` 、`./exps/fixed_cameras/elephant-eval`、 `./exps/fixed_cameras/frog-eval`、 `./exps/fixed_cameras/squirrel-eval`,  respectively.



## Usage

### Train

To train the model, after configuring the configuration file`./coode/conf/fixed_cameras.conf`, run the following commands:

```
cd ./coode
python training/exp_runner.py --conf='./confs/fixed_cameras.conf'
```

The folder that contains the trained results is named with timestamp under `./exps/fixed_cameras/` directory.

### Evaluation on pretrained models

To evaluate the pretrained models, after configuring the configuration file`./coode/conf/fixed_cameras_eval.conf`, run the following commands:

```
cd ./coode
python training/exp_runner_eval.py --conf='./confs/fixed_cameras_eval.conf' --timestamp=$OBJECT_NAME$
```

where `$OBJECT_NAME$` is one of the following values: `cat-eval,elephant-eval,frog-eval,squirrel-eval`.  The evaluation results will store in the directory under `./exps/fixed_cameras/` and is named with the latest timestamp. 

## Citation

If you find our work useful in your research, please consider citing:

```
@article{shao2022polarimetric,
  title={Polarimetric Inverse Rendering for Transparent Shapes Reconstruction},
  author={Shao, Mingqi and Xia, Chongkun and Duan, Dongxu and Wang, Xueqian},
  journal={arXiv preprint arXiv:2208.11836},
  year={2022}
}
```

## Acknowledgement

In this work we use parts of the implementations of the following work:

- [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance](https://github.com/lioryariv/idr)

We thank the authors for open sourcing their method.
