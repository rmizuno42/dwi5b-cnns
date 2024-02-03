# Dataset

We assume the dataset structure is the following.

```
dataset_root
├── tumor_id_0
│   ├── slice_number_0.pkl
│   ├── slice_number_1.pkl
│   └── ...
├── tumor_id_1
└── ...
```

where slice_number_N is an integer.
A slice_number_N.pkl file is a List object where the first element is a numpy array of the DWI 5b data for the slice, and the second element is a string representing the class of the slice. The shape of the DWI 5b data is (5, height, width). the class is 'M' if the slice contains a malignant tumor, 'B' if it contains a benign tumor, and 'N' otherwise.

# 10-fold cross validation

```
python3 train_2d.py  --lr 0.03 --net simple --target dataset --mixup --affine_noise --elastic --logdir logs_2d
```
```
python3 train_3d.py --lr 0.03 --net simple --target dataset_3d --mixup   --affine_noise --elastic --logdir logs_3d
```

Options
- --lr : learning rate
- --target : dataset root path 
- --net : network structure to train
    - simple : Small 2D CNN or Small 3D CNN
    - resnet : ResNet18
    - efficient : EfficientNet-B0
- --mixup : use Mixup
- --affine_noise : use random affine transform and random noise 
- --elastic : use erastic deformation
- --logdir : output directory name

Please refer to the help or the code for other options.

# Test

```
python3 eval_2d.py  --weight_dir logs_2d --test_target dataset_test --output eval_output_2d
```
```
python3 eval_3d.py  --weight_dir logs_3d --test_target dataset_3d_test --output eval_output_3d
```

# Collecting labels and predictions for each data
```
python3 collect_results_2d.py --target_type validation --target logs_2d 
```
```
python3 collect_results_2d.py --target_type eval --target eval_output_2d
```
```
python3 collect_results_3d.py --target_type validation --target logs_3d
```
```
python3 collect_results_3d.py --target_type eval --target eval_3d_output_3d
```