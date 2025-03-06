#!/usr/bin/env python3
"""
File to run inference on the SoccerNet Action Spotting data
"""

#Standard imports
import argparse
import os
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import wandb
import sys
import json

#Local imports
from config.config import Config
from util.io import load_json, load_text
from util.dataset import load_classes
from model.model import TDEEDModel
from util.eval import evaluate
from SoccerNet.Evaluation.ActionSpotting import evaluate as evaluate_SN
from dataset.frame import ActionSpotVideoDataset


#Constants
EVAL_SPLITS = ['test']
STRIDE_SN = 12


def main(*, model: str, acc_grad_iter: int = 1, seed: int = 1):

    initial_time = time.time()

    #Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load config
    config_path = model.split('_')[0] + '/' + model + '.json'
    config = load_json(os.path.join('config', config_path))
    config = Config.model_validate(config)
    config.save_dir = config.save_dir + '/' + model

    LABELS_SN_PATH = load_text(os.path.join('data', config.dataset, 'labels_path.txt'))[0]

    assert config.batch_size % acc_grad_iter == 0

    # initialize wandb
    wandb.login()
    wandb.init(config = config, dir = config.save_dir + '/wandb_logs', project = 'ExtendTDEED', name = model + '-' + str(seed))
    
    classes = load_classes(os.path.join('data', config.dataset, 'class.txt'))

    # Model
    model = TDEEDModel(device='cuda', config=config)

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), 'checkpoints', config.model.split('_')[0], config.model, 'checkpoint_best.pt')))

    for split in EVAL_SPLITS:
        split_path = os.path.join(
            'data', config.dataset, '{}.json'.format(split))

        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes, split_path, config.frame_dir, config.modality,
                config.clip_len, overlap_len = config.clip_len // 2,
                stride = STRIDE_SN, dataset = config.dataset)

            pred_file = None
            if config.save_dir is not None:
                pred_file = os.path.join(
                    config.save_dir, 'pred-{}'.format(split))

            mAPs, tolerances = evaluate(model, split_data, split.upper(), classes, pred_file, printed = True, 
                        test = True, augment = False)
            
            if split != 'challenge':
                for i in range(len(mAPs)):
                    wandb.log({'test/mAP@' + str(tolerances[i]): mAPs[i]})
                    wandb.summary['test/mAP@' + str(tolerances[i])] = mAPs[i]

                if config.dataset == 'soccernet':

                    with open(pred_file + ".json", 'r') as f:
                        preds = json.load(f)

                    predictions_path = "/".join(pred_file.split('/')[:-1])

                    for pred in preds:
                        print("video: ", pred['video'])
                        print("length of events: ", len(pred["events"]))
                        
                        # Create full directory path including all subdirectories
                        full_pred_path = os.path.join(predictions_path, pred['video']).replace("/half1", "").replace("/half2", "")
                        os.makedirs(full_pred_path, exist_ok=True)
                        
                        # Save the results file
                        results_path = os.path.join(full_pred_path, "results_spotting.json")
                        with open(results_path, 'w') as f:
                            json.dump(pred["events"], f)
                        print(f"Saved results to: {results_path}")  # Add debug print

                    results = evaluate_SN(
                        LABELS_SN_PATH, predictions_path, 
                        split = split, prediction_file = "results_spotting.json", version = 2, 
                        metric = "tight"
                    )

                    print('Tight aMAP: ', results['a_mAP'] * 100)
                    print('Tight aMAP per class: ', results['a_mAP_per_class'])

                    wandb.log({'test/mAP': results['a_mAP'] * 100})
                    wandb.summary['test/mAP'] = results['a_mAP'] * 100

                    for j in range(len(classes)):
                        wandb.log({'test/classes/mAP@' + list(classes.keys())[j]: results['a_mAP_per_class'][j] * 100})

    print('CORRECTLY FINISHED INFERENCE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1, help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    main(
        model=args.model,
        acc_grad_iter=args.acc_grad_iter,
        seed=args.seed
    )