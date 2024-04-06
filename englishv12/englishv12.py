# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CoNLL2012 shared task data based on OntoNotes 5.0"""

import glob
import os
from collections import defaultdict
from typing import DefaultDict, Iterator, List, Optional, Tuple

import datasets


_CITATION = """\
@inproceedings{pradhan-etal-2013-towards,
    title = "Towards Robust Linguistic Analysis using {O}nto{N}otes",
    author = {Pradhan, Sameer  and
      Moschitti, Alessandro  and
      Xue, Nianwen  and
      Ng, Hwee Tou  and
      Bj{\"o}rkelund, Anders  and
      Uryupina, Olga  and
      Zhang, Yuchen  and
      Zhong, Zhi},
    booktitle = "Proceedings of the Seventeenth Conference on Computational Natural Language Learning",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-3516",
    pages = "143--152",
}

Ralph Weischedel, Martha Palmer, Mitchell Marcus, Eduard Hovy, Sameer Pradhan, \
Lance Ramshaw, Nianwen Xue, Ann Taylor, Jeff Kaufman, Michelle Franchini, \
Mohammed El-Bachouti, Robert Belvin, Ann Houston. \
OntoNotes Release 5.0 LDC2013T19. \
Web Download. Philadelphia: Linguistic Data Consortium, 2013.
"""

_DESCRIPTION = """\
OntoNotes v5.0 is the final version of OntoNotes corpus, and is a large-scale, multi-genre,
multilingual corpus manually annotated with syntactic, semantic and discourse information.

This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL-2012 shared task.
It includes v4 train/dev and v9 test data for English/Chinese/Arabic and corrected version v12 train/dev/test data (English only).

The source of data is the Mendeley Data repo [ontonotes-conll2012](https://data.mendeley.com/datasets/zmycy7t9h9), which seems to be as the same as the official data, but users should use this dataset on their own responsibility.

See also summaries from paperwithcode, [OntoNotes 5.0](https://paperswithcode.com/dataset/ontonotes-5-0) and [CoNLL-2012](https://paperswithcode.com/dataset/conll-2012-1)

For more detailed info of the dataset like annotation, tag set, etc., you can refer to the documents in the Mendeley repo mentioned above.
"""

_URL = "https://data.mendeley.com/public-files/datasets/zmycy7t9h9/files/b078e1c4-f7a4-4427-be7f-9389967831ef/file_downloaded"


class Englishv12Config(datasets.BuilderConfig):
    """BuilderConfig for the CoNLL formatted OntoNotes dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for the CoNLL formatted OntoNotes dataset.

        Args:
          language: string, one of the language {"english", "chinese", "arabic"} .
          conll_version: string, "v4" or "v12". Note there is only English v12.
          **kwargs: keyword arguments forwarded to super.
        """
        language = "english"
        conll_version = "v12"
        super(Englishv12Config, self).__init__(
            name=f"{language}_{conll_version}",
            description=f"{conll_version} of CoNLL formatted OntoNotes dataset for {language}.",
            version=datasets.Version("1.0.0"),  # hf dataset script version
            **kwargs,
        )
        self.language = language
        self.conll_version = conll_version


class Englishv12(datasets.GeneratorBasedBuilder):
    """The CoNLL formatted OntoNotes dataset."""

    BUILDER_CONFIGS = [
        Englishv12Config(
        )
    ]

    def _info(self):
        lang = self.config.language
        conll_version = self.config.conll_version

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(num_classes=37, names=_NAMED_ENTITY_TAGS)
                    )
                }
            ),
            homepage="https://conll.cemantix.org/2012/introduction.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.language
        conll_version = self.config.conll_version
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, f"conll-2012/{conll_version}/data")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"conll_files_directory": os.path.join(data_dir, f"train/data/{lang}")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"conll_files_directory": os.path.join(data_dir, f"development/data/{lang}")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"conll_files_directory": os.path.join(data_dir, f"test/data/{lang}")},
            ),
        ]

    def _generate_examples(self, conll_files_directory):
        conll_files = sorted(glob.glob(os.path.join(conll_files_directory, "**/*gold_conll"), recursive=True))
        guid = 0
        for idx, conll_file in enumerate(conll_files):
            sentences = []
            for sent in Ontonotes().sentence_iterator(conll_file):
                yield guid, {       
                        "tokens": sent.words,            
                        "ner_tags": sent.named_entities,
                    }
                guid += 1
                


# --------------------------------------------------------------------------------------------------------
# Tag set
_NAMED_ENTITY_TAGS = [
    "O",  # out of named entity
    "B-PERSON",
    "I-PERSON",
    "B-NORP",
    "I-NORP",
    "B-FAC",  # FACILITY
    "I-FAC",
    "B-ORG",  # ORGANIZATION
    "I-ORG",
    "B-GPE",
    "I-GPE",
    "B-LOC",
    "I-LOC",
    "B-PRODUCT",
    "I-PRODUCT",
    "B-DATE",
    "I-DATE",
    "B-TIME",
    "I-TIME",
    "B-PERCENT",
    "I-PERCENT",
    "B-MONEY",
    "I-MONEY",
    "B-QUANTITY",
    "I-QUANTITY",
    "B-ORDINAL",
    "I-ORDINAL",
    "B-CARDINAL",
    "I-CARDINAL",
    "B-EVENT",
    "I-EVENT",
    "B-WORK_OF_ART",
    "I-WORK_OF_ART",
    "B-LAW",
    "I-LAW",
    "B-LANGUAGE",
    "I-LANGUAGE",
]

# --------------------------------------------------------------------------------------------------------
# The CoNLL(2012) file reader
# Modified the original code to get rid of extra package dependency.
# Original code: https://github.com/allenai/allennlp-models/blob/main/allennlp_models/common/ontonotes.py


class OntonotesSentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.
    # Parameters
    document_id : `str`
        This is a variation on the document filename
    sentence_id : `int`
        The integer ID of the sentence within a document.
    words : `List[str]`
        This is the tokens as segmented/tokenized in the bank.
    named_entities : `List[str]`
        The BIO tags for named entities in the sentence.
    """

    def __init__(
        self,
        document_id: str,
        sentence_id: int,
        words: List[str],
        named_entities: List[str],
    ) -> None:

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.named_entities = named_entities



class Ontonotes:
    """
    This `DatasetReader` is designed to read in the English OntoNotes v5.0 data
    in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
    Reader, you must follow the instructions provided [here (v12 release):]
    (https://cemantix.org/data/ontonotes.html), which will allow you to download
    the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
    obtained from LDC.
    Once you have run the scripts on the extracted data, you will have a folder
    structured as follows:
    ```
    conll-formatted-ontonotes-5.0/
     ── data
       ├── development
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       ├── test
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       └── train
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
    ```
    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).
    The data has the following format, ordered by column.
    1.  Document ID : `str`
        This is a variation on the document filename
    2.  Part number : `int`
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3.  Word number : `int`
        This is the word index of the word in that sentence.
    4.  Word : `str`
        This is the token as segmented/tokenized in the Treebank. Initially the `*_skel` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    11. Named Entities : `str`
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an `*`.
    """

    def dataset_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        for root, _, files in list(os.walk(file_path)):
            for data_file in sorted(files):
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntonotesSentence]]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with open(file_path, "r", encoding="utf8") as open_file:
            conll_rows = []
            document: List[OntonotesSentence] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        document_id: str = None
        sentence_id: int = None
        # The words in the sentence.
        sentence: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        for index, row in enumerate(conll_rows):
            conll_components = row.split()

            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]
            sentence.append(word)
            self._process_span_annotations_for_word(conll_components[10:-1], span_labels, current_span_labels)

        named_entities = span_labels[0]

        return OntonotesSentence(
            document_id,
            sentence_id,
            sentence,
            named_entities,
        )

    @staticmethod
    def _process_span_annotations_for_word(
        annotations: List[str],
        span_labels: List[List[str]],
        current_span_labels: List[Optional[str]],
    ) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.
        # Parameters
        annotations : `List[str]`
            A list of labels to compute BIO tags for.
        span_labels : `List[List[str]]`
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : `List[Optional[str]]`
            The currently open span per annotation type, or `None` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None
