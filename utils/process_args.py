import argparse

def _process_args():

    parser = argparse.ArgumentParser(description='Configurations for SurvTransformer Survival Prediction Training')

    parser.add_argument('--study', type=str, default='tcga_stad', help='study type')
    parser.add_argument('--sig', type=str, default='our')
    parser.add_argument('--task', type=str, default='survival')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--need_test', default=False, help='test')


    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--num_patches', type=int, default=4096, help='Number of fixed patches sampled')
    parser.add_argument('--label_col', type=str, default="survival_months", help='survival_months')
    parser.add_argument("--wsi_projection_dim", type=int, default=1)

    parser.add_argument('--k', type=int, default=5, help='number of folds')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='./splits/5foldcv/tcga_stad', help='split')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')

    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=666, help='random seed for reproducible experiment')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="pathcell regularization type [None, pathcell]")
    parser.add_argument('--lambda_reg', type=float, default=1e-5, help='L1-Regularization Strength')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, default='nll_surv', help='survival loss function')
    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='Enable early stopping')
    
    #---> model related
    parser.add_argument('--fusion', type=str, default="None", help="None, concat, bilinear")
    parser.add_argument('--modality', type=str, default="SurvTransformer")
    parser.add_argument('--encoding_dim', type=int, default=1024, help='WSI encoding dim')

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args