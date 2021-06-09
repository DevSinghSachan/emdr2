import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.global_vars import set_global_variables
from tasks.openqa.e2eqa.async_indexer import initialize_and_run_async_megatron


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True, help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetuning epochs. 0 results in evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names for training.')
    group.add_argument('--valid-data', nargs='*', default=None, help='path(s) to the validation data.')
    group.add_argument('--test-data', nargs='*', default=None, help='path(s) to the test data.')
    group.add_argument('--beam-size', default=1, type=int,
                       help='Beam size to use for decoding. A beam size of 1 corresponds to greedy search')
    group.add_argument('--max-decode-len', default=512, type=int,
                       help='maximum sequence length to generate at the decoder.')
    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch size). Global batch size is local'
                             ' batch size times data parallel size.')

    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument("--val-av-rank-hard-neg", type=int, default=30,
                        help="Av.rank validation: how many hard negatives to take from each question pool")
    group.add_argument("--val-av-rank-other-neg", type=int, default=30,
                        help="Av.rank validation: how many 'other' negatives to take from each question pool")
    group.add_argument("--train-with-neg", action='store_true',
                       help="Whether to use negative examples during model training")
    group.add_argument("--train-hard-neg", type=int, default=0,
                       help="Number of hard negative exmaples to use during training")
    return parser


if __name__ == '__main__':

    set_global_variables(extra_args_provider=get_tasks_args,
                         args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
                         ignore_unknown_args=False)
    args = get_args()

    if args.async_indexer:
        initialize_and_run_async_megatron()
    else:
        initialize_megatron()

    if args.task == "RETRIEVER":
        from tasks.openqa.dense_retriever.run import main
    elif args.task == "OPENQA":
        from tasks.openqa.e2eqa.run import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    main()
