import glob
from megatron.initialize import initialize_megatron, get_args
from megatron.global_vars import set_global_variables
from tasks.openqa.dense_retriever.evaluation.evaluate import OpenRetrievalEvaluator


def main():
    set_global_variables(extra_args_provider=None,
                         args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
                         ignore_unknown_args=False)
    initialize_megatron()

    args = get_args()
    evaluator = OpenRetrievalEvaluator()

    if args.qa_file_dev is not None:
        if args.glob:
            all_files = glob.glob(args.qa_file_dev)
            for file in all_files:
                evaluator.evaluate(file, "DEV")
        else:
            evaluator.evaluate(args.qa_file_dev, "DEV")

    if args.qa_file_test is not None:
        if args.glob:
            all_files = glob.glob(args.qa_file_test)
            for file in all_files:
                evaluator.evaluate(file, "TEST")
        else:
            evaluator.evaluate(args.qa_file_test, "TEST")

    if args.qa_file_train is not None:
        evaluator.evaluate(args.qa_file_train, "TRAIN")


if __name__ == "__main__":
    main()

