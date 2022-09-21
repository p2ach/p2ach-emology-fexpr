import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import FaceExpr
from train_options import ToolOptions
from lexre import ExpRecognition


# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt"
# }


def run_train(args):
    model = ExpRecognition()
    model.prepare_devices(args.gpu_ids, args.landmark_num,args.chns)

    if args.mode == 'train':
        model.load_train_data(args.train_path, args.batch_size, args.num_workers)
        model.load_validation_data(args.validation_path, args.batch_size, args.num_workers)
        # model.load_test_data(opt.test_image_path, opt.num_workers)
        model.prepare_tool(args.start_lr, args.learning_rate_decay_start, args.total_epoch, args.model_path, \
                           args.beta, args.margin_1, args.margin_2, args.relabel_epoch)

        for epoch in range(1, args.total_epoch + 1):
            model.train(epoch)
            model.validation(epoch)

if __name__ == '__main__':
    args = ToolOptions().parse()
    run_train(args)


