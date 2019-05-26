import os

#os.system('python cifar.py -a dia_preresnet --train-batch 128 --dataset cifar100 --depth 164 --block-name bottleneck --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/preresnet-164-dia-rate4')
#os.system('python cifar.py -a dia_wrn --dataset cifar100 --lr 0.1 --depth 52 --widen-factor 4 --drop 0.3 --epochs 200 --schedule 80 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar100/WRN-52-4-drop-dia-rate4')
#os.system('python cifar.py -a dia_resnext --dataset cifar100 --depth 101 --cardinality 8 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-8x32d-101-dia-rate4 --schedule 150 225 --wd 5e-4 --gamma 0.1')
os.system('python cifar.py -a dia_resnet --train-batch 128 --dataset cifar100 --depth 164 --block-name bottleneck --epochs 180 --schedule 60 120 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-164-dia-rate4')
