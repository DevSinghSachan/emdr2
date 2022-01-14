import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import argparse
import torch
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name, ensure_directory_exists


def main(args):
    tracker_filename = get_checkpoint_tracker_filename(args.load)

    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)

    checkpoint_name = get_checkpoint_name(args.load, iteration, release=False, mp_rank=0)
    state_dict = torch.load(checkpoint_name, map_location='cpu')

    model_state_dict = state_dict.pop('model')

    if args.submodel_name == "retriever":
        ret_weights = model_state_dict.pop('retriever/biencoder_model')
        ret_state_dict = {'model': ret_weights}
        ret_state_dict.update(state_dict)
        save_state_dict = ret_state_dict
    elif args.submodel_name == "encoder":
        encoder_weights = model_state_dict.pop('encoder/t5_model')
        enc_state_dict = {'model': encoder_weights}
        enc_state_dict.update(state_dict)
        save_state_dict = enc_state_dict
    else:
        raise AssertionError("invalid submodel name to be save")

    save_checkpoint_name = get_checkpoint_name(args.save, iteration, release=False, mp_rank=0)
    ensure_directory_exists(save_checkpoint_name)
    torch.save(save_state_dict, save_checkpoint_name)
    print('  successfully saved {}'.format(save_checkpoint_name))
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    with open(tracker_filename, 'w') as f:
        f.write(str(iteration))


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='save separate EMDR2 models')
    group.add_argument('--load', type=str, required=True, help='EMDR2 model path')
    group.add_argument('--save', type=str, required=True, help='path to save model')
    group.add_argument('--submodel-name', type=str, required=True, choices=['encoder', 'retriever'],
                       help='sub model of EMDR2 to save')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
