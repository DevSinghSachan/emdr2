import sys
sys.path.append('../')

from megatron.indexer_emdr2 import IndexBuilder
from megatron.global_vars import set_global_variables
from megatron.initialize import initialize_megatron


def main():
    set_global_variables(extra_args_provider=None,
                         args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
                         ignore_unknown_args=False)
    initialize_megatron()
    
    index_builder = IndexBuilder()
    index_builder.build_and_save_index()


if __name__ == "__main__":
    main()
