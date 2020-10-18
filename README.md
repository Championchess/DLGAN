### Paper

 Guanqi Zhan, Yihao Zhao, Bingchan Zhao, Haoqi Yuan, Baoquan Chen, Hao Dong, ["DLGAN: Disentangling Label-Specific Fine-Grained Features for Image Manipulation"](https://arxiv.org/abs/1911.09943)


## File Structure

This repository provides data and code as follows.


```
    celebA/						# place celebA dataset here 
    DukeMTMC/					# place DukeMTMC dataset here 
    models/
    	prefix/					# saved models
    results/
    	prefix/					# results at different training steps
    
    utils.py 					# utilization functions
    train.py 					# main file to train
    data.py 					# data loader
    models.py 					# PyTorch
    config.py 					# training configuration

    DLGAN.yaml					# Conda running environment
    		
```

## Dependencies

To create conda environment and install dependencies:

`conda env create -f DLGAN.yaml`

## Quick Start
Download the [trained models](???) and place them under 'models/'.

## Train the model: 
For [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, you should download and place all the images under 'celebA/img_align_celebA/', place 'list_attr_celeba.txt' file under 'celebA/'. Then, run
`python train.py --dataset celebA --batch_size 32`

For [DukeMTMC-reID](https://github.com/vana77/DukeMTMC-attribute) dataset, download and place all the images under 'DukeMTMC/bounding_box_train/', place 'duke_attribute.mat' file under 'DukeMTMC/'. Then, run
`python train.py --dataset DukeMTMC --batch_size 32`


## Citation

If you find this code useful for your research, please cite our paper:

```
@misc{zhan2020dlgan,
      title={DLGAN: Disentangling Label-Specific Fine-Grained Features for Image Manipulation}, 
      author={Guanqi Zhan and Yihao Zhao and Bingchan Zhao and Haoqi Yuan and Baoquan Chen and Hao Dong},
      year={2020},
      eprint={1911.09943},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
