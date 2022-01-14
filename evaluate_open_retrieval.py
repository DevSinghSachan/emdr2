from megatron.initialize import initialize_megatron, get_args
from tasks.openqa.dense_retriever.evaluation.evaluate import OpenRetrievalEvaluator


def main():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    evaluator = OpenRetrievalEvaluator()

    if args.qa_file_dev is not None:
        evaluator.evaluate(args.qa_file_dev, "DEV")

    if args.qa_file_test is not None:
        evaluator.evaluate(args.qa_file_test, "TEST")


if __name__ == "__main__":
    main()

