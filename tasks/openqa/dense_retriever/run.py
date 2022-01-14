from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.model.dualencoder_model import dualencoder_model_provider
from tasks.openqa.dense_retriever.train_dense_retriever import accuracy_func_provider
from tasks.openqa.dense_retriever.train_dense_retriever import train


def dense_retriever(dataset_cls):

    def train_valid_datasets_provider():
        args = get_args()
        tokenizer = get_tokenizer()

        train_dataset = dataset_cls("training",
                                    args.train_data,
                                    tokenizer,
                                    args.seq_length_ret,
                                    evaluate=False)
        valid_dataset = dataset_cls("validation",
                                    args.valid_data,
                                    tokenizer,
                                    args.seq_length_ret,
                                    evaluate=True)
        return train_dataset, valid_dataset

    def model_provider():
        args = get_args()
        print_rank_0('building retriever model for {} ...'.format(args.task))
        model = dualencoder_model_provider(only_context_model=False,
                                           only_query_model=False)
        return model

    def single_dataset_provider(datapath):
        args = get_args()
        tokenizer = get_tokenizer()
        name_from_datapath = datapath[0].split('/')[-1].split('.')[0]

        return dataset_cls(name_from_datapath,
                           datapath,
                           tokenizer,
                           args.seq_length_ret,
                           evaluate=True)

    def distributed_metrics_func_provider(datapath):
        return accuracy_func_provider(single_dataset_provider, datapath)


    train(train_valid_datasets_provider,
          model_provider,
          end_of_epoch_callback_provider=distributed_metrics_func_provider)


def main():
    args = get_args()

    if args.task == 'RETRIEVER':
        from tasks.openqa.dense_retriever.train_data_utils import Dataset as dataset_cls
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    dense_retriever(dataset_cls)
