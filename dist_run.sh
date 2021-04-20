clear
python -m torch.distributed.launch --nproc_per_node=4 main.py --name baseline --dataset_root ../../DataSet/COCO