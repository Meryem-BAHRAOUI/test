
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------- model ----------
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # ---------- data ----------
        self.data_dir = "/content/YOLOX/datasets/ui_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # ---------- input ----------
        self.input_size = (1280, 1280)
        self.test_size = (1280, 1280)

        # ---------- training ----------
        self.max_epoch = 150
        self.data_num_workers = 2
        self.eval_interval = 1
        self.print_interval = 20

        # Pour dataset UI : réduire fortement les augmentations
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.8, 1.2)
        self.shear = 0.0
        self.enable_mixup = False

        # epochs finales sans augmentation forte
        self.no_aug_epochs = 10

        # LR / opti
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
