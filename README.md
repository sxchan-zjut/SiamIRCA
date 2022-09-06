# SiamIRCA

This project hosts the code for implementing the SiamIRCA algorithm for visual tracking, as presented in our paper: 
## Siamese Implicit Region Proposal Network With Compound Attention for Visual Tracking [here](https://ieeexplore.ieee.org/document/9709213)


## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SiamIRCA

### Add SiamIRCA to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

### Download models

Download models in [here](https://) and put the `model.pth` in the correct directory in experiments

### Webcam demo

```bash
python tools/demo.py \
    --config config.yaml \
    --snapshot xxxx/xxxx/ #model path
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd tools
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker


``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.

## License

This project is released under the Apache 2.0 license. 
