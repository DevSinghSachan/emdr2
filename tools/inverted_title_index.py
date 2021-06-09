import csv
import argparse
from collections import defaultdict
import bisect

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import print_rank_0


class WikiTitleDocMap():
    """Open Retrieval Evidence dataset class."""

    def __init__(self, datapath):
        print_rank_0(datapath)
        # self.max_seq_length = args.seq_length_ret
        self.title2docs, self.docid2title = self.process_wikipedia(datapath)

    def get_neighbour_paragraphs(self, doc_id):
        title = self.docid2title[doc_id]
        doc_row = self.title2docs[title]

        # 'Locate the leftmost value exactly equal to doc_row'
        i = bisect.bisect_left(doc_row, doc_id)
        if i != len(doc_row) and doc_row[i] == doc_id:
            # First condition: if the doc id is the first element in the list
            # Automatically takes care of variable sized lists
            if i == 0:
                return doc_row[i: i+3], 0
            elif i == len(doc_row) - 1:
                return doc_row[i-2: i+1], -1
            else:
                return doc_row[i-1: i+2], 1
        raise ValueError

    @staticmethod
    def process_wikipedia(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        total = 0

        title2ids = defaultdict(list)
        docid2title = {}

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                title2ids[title].append(doc_id)
                assert doc_id not in docid2title
                docid2title[doc_id] = title

                total += 1
                if total % 100000 == 0:
                    print_rank_0('  > processed {} rows so far ...'.format(total))

        return title2ids, docid2title


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to the input Wikipedia file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    obj = WikiTitleDocMap(args.input)
    print(obj.get_neighbour_paragraphs(5))
