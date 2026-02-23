from __future__ import print_function
from config.config_linear import parse_option
from training_linear.training_one_epoch_ckpt import main
from training_linear.training_one_epoch_fusion import main_supervised_fusion
from training_linear.training_one_epoch_supervised import main_supervised
from training_linear.training_one_epoch_supervised_multilabel import main_supervised_multilabel
from training_linear.training_one_epoch_fusion_multilabel import main_supervised_multilabel_fusion
from training_linear.training_one_epoch_ckpt_multi import main_multilabel
from training_linear.training_one_epoch_Ford import main_supervised_Ford
from training_linear.training_one_epoch_ckpt_ford import main_Ford
from training_linear.training_one_epoch_ckpt_bce import main_bce
from training_linear.training_one_epoch_chest_linear import main_chest
from training_linear.training_one_epoch_chest_linaer_supervised import main_chest_super
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass







if __name__ == '__main__':
    opt = parse_option()
    if(opt.dataset == 'Ford_Region' and opt.super == 1):
        main_supervised_Ford()
    elif(opt.dataset == 'Ford_Region' and opt.super == 0):
        main_Ford()
    if(opt.dataset == 'covid_kaggle' and opt.super == 0):
        main_chest()
    elif(opt.dataset == 'covid_kaggle' and opt.super == 1):
        main_chest_super()
    else:
        if(opt.super==1 and opt.multi ==0 ):
            print('Supervised')
            main_supervised()
        elif(opt.super == 3 and opt.multi == 1):
            print("AUROC Training")
            main_bce()
        elif(opt.super == 2 and opt.multi == 0):
            print('Fusion Supervised')
            main_supervised_fusion()
        elif(opt.multi == 1 and opt.super == 1):
            main_supervised_multilabel()
        elif(opt.multi == 1 and opt.super == 2):
            main_supervised_multilabel_fusion()
        elif(opt.multi == 1 and opt.super == 0):
            print('Using CKPT')
            main_multilabel()
        else:
            print('Using CKPT')
            main()