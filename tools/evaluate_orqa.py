
from megatron.initialize import initialize_megatron

from tasks.openqa.dense_retriever.evaluation.evaluate import OpenRetrievalEvaluator


def main():
    """
    """

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    evaluator = OpenRetrievalEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()

