from __future__ import absolute_import, division, print_function
import json
import logging
import os
import glob
import re
from io import open
import torch
import tqdm
from typing import List

from transformers.tokenization_bert import PreTrainedTokenizer

from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def load_and_cache_several_examples(args, tasks, tokenizer, evaluate=False, test=False, no_para=True):
    all_features = []
    for task in tasks:
        if 'Add' not in task:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            args.data_dir = os.path.join(args.raw_data_dir, task)
            processor = processors[task.lower()]()
            # Load data features from cache or dataset file
            if evaluate:
                cached_mode = 'dev'
            elif test:
                cached_mode = 'test'
            else:
                cached_mode = 'train'
            assert (evaluate == True and test == True) == False
            cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
                cached_mode,
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length),
                str(task)))
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                label_list = processor.get_labels()
                if evaluate:
                    examples = processor.get_dev_examples(args.data_dir)
                elif test:
                    examples = processor.get_test_examples(args.data_dir)
                else:
                    examples = processor.get_train_examples(args.data_dir)
                logger.info("Total number: %s", str(len(examples)))
                features = convert_examples_to_features(
                    examples,
                    label_list,
                    args.max_seq_length,
                    tokenizer,
                    pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                    no_para=no_para
                )
                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
        else:
            logger.info("Evaluate on {}!".format(task))
            task, adv_type = task.split('-')
            args.data_dir = os.path.join(args.raw_data_dir, task)
            processor = processors[task.lower()]()
            examples = processor.get_adv_examples(args.data_dir, adv_type=adv_type)
            label_list = processor.get_labels()
            logger.info("Total number: %s", str(len(examples)))
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                no_para=no_para
            )
        all_features.extend(features)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    features = all_features
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    if no_para:
        all_input_ids_np = torch.tensor(select_field(features, 'input_ids_np'), dtype=torch.long)
        all_input_mask_np = torch.tensor(select_field(features, 'input_mask_np'), dtype=torch.long)
        all_segment_ids_np = torch.tensor(select_field(features, 'segment_ids_np'), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_input_ids_np, all_input_mask_np, all_segment_ids_np,
                            all_label_ids)
    return dataset


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False, no_para=True):
    if 'Add' not in task:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        args.data_dir = os.path.join(args.raw_data_dir, task)
        processor = processors[task.lower()]()
        # Load data features from cache or dataset file
        if evaluate:
            cached_mode = 'dev'
        elif test:
            cached_mode = 'test'
        else:
            cached_mode = 'train'
        assert (evaluate == True and test == True) == False
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir)
            elif test:
                examples = processor.get_test_examples(args.data_dir)
            else:
                examples = processor.get_train_examples(args.data_dir)
            logger.info("Total number: %s", str(len(examples)))
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                no_para=no_para
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
    else:
        logger.info("Evaluate on {}!".format(task))
        task, adv_type = task.split('-')
        args.data_dir = os.path.join(args.raw_data_dir, task)
        processor = processors[task.lower()]()
        examples = processor.get_adv_examples(args.data_dir, adv_type=adv_type)
        label_list = processor.get_labels()
        logger.info("Total number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            no_para=no_para
        )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    if no_para:
        all_input_ids_np = torch.tensor(select_field(features, 'input_ids_np'), dtype=torch.long)
        all_input_mask_np = torch.tensor(select_field(features, 'input_mask_np'), dtype=torch.long)
        all_segment_ids_np = torch.tensor(select_field(features, 'segment_ids_np'), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_input_ids_np, all_input_mask_np, all_segment_ids_np,
                            all_label_ids)
    return dataset


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'input_ids_np': input_ids_np,
                'input_mask_np': input_mask_np,
                'segment_ids_np': segment_ids_np,
            }
            for input_ids, input_mask, segment_ids,
                input_ids_np, input_mask_np, segment_ids_np in choices_features
        ]
        self.label = label

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_adv_examples(self, data_dir, adv_type):
        path = os.path.join(data_dir, adv_type+'.pkl')
        with open(path, 'rb') as f:
            examples = torch.load(f)
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article], # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        return examples

class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        #There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id = id,
                        question=question,
                        contexts=[options[0]["para"].replace("_", ""), options[1]["para"].replace("_", ""),
                                  options[2]["para"].replace("_", ""), options[3]["para"].replace("_", "")],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples

class MctestProcessor(DataProcessor):
    """Processor for the MCTest data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_file(data_dir, "train"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_file(data_dir, "dev"), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_file(data_dir, "test"), "test")

    def get_adv_examples(self, data_dir, adv_type):
        path = os.path.join(data_dir, adv_type + '.pkl')
        with open(path, 'rb') as f:
            examples = torch.load(f)
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_file(self, data_dir, set_name):
        context_160_file = 'mc160.' + set_name + '.tsv'
        context_500_file = 'mc500.' + set_name + '.tsv'
        answer_160_file = 'mc160.' + set_name + '.ans'
        answer_500_file = 'mc500.' + set_name + '.ans'
        with open(os.path.join(data_dir, context_160_file)) as f:
            context_160 = f.read()
        with open(os.path.join(data_dir, context_500_file)) as f:
            context_500 = f.read()
        with open(os.path.join(data_dir, answer_160_file)) as f:
            answer_160 = f.read()
        with open(os.path.join(data_dir, answer_500_file)) as f:
            answer_500 = f.read()
        context = (context_160, context_500)
        answer = (answer_160, answer_500)
        return (context, answer)

    def _create_examples(self, context_answer, set_name):
        raw_context, raw_answer = context_answer[0], context_answer[1]
        raw_context_160, raw_context_500 = raw_context[0], raw_context[1]
        raw_answer_160, raw_answer_500 = raw_answer[0], raw_answer[1]
        answer_160 = [ord(option)-ord('A') for option in raw_answer_160 if option in ['A', 'B', 'C', 'D']]
        context_160 = raw_context_160.split('\n')[:-1]
        answer_500 = [ord(option)-ord('A') for option in raw_answer_500 if option in ['A', 'B', 'C', 'D']]
        context_500 = raw_context_500.split('\n')[:-1]
        context = context_160+context_500
        answer = answer_160+answer_500
        idx = 0
        examples = []
        for i, sample in enumerate(context):
            elements = sample.split('\t')
            passage = elements[2] # remove title newlines and tabs
            passage = re.sub(r'\\newline', '\n', passage)
            passage = re.sub(r'\\tab', '\t', passage)
            for j in range(4):
                question_elements = elements[3 + 5 * j:3 + 5 * (j + 1)]  # get question elements
                qtype, qtext = question_elements[0].split(': ')  # get question type and text
                options = [text for text in question_elements[1:5]]  # get answers
                truth = answer[idx]  # get correct answer (from answer data)
                idx += 1
                examples.append(
                    InputExample(
                        example_id=idx,
                        question=qtext,
                        contexts=[passage, passage, passage, passage],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        assert len(examples) == len(answer)
        return examples

class Semeval2018Processor(DataProcessor):
    """Processor for the SemEval 2018 Task 11 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train-data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev-data.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test-data.json")), "test")

    def get_adv_examples(self, data_dir, adv_type):
        path = os.path.join(data_dir, adv_type + '.pkl')
        with open(path, 'rb') as f:
            examples = torch.load(f)
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    def _create_examples(self, data, type):
        """Creates examples for the training and dev sets."""
        dataset = data['data']['instance']
        idx = 0
        examples = []
        corrupted_sample = 0
        for i in dataset:
            passage = i['text']
            try:
                if isinstance(i['questions']['question'], list):
                    for question in i['questions']['question']:
                        query = question['@text']
                        options = [item['@text'] for item in question['answer']]
                        truth = 0 if question['answer'][0]['@correct'] == 'True' else 1
                        examples.append(
                            InputExample(
                                example_id=idx,
                                question=query,
                                contexts=[passage, passage],  # this is not efficient but convenient
                                endings=[options[0], options[1]],
                                label=truth))
                        idx += 1
                else:
                    question = i['questions']['question']
                    query = question['@text']
                    options = [item['@text'] for item in question['answer']]
                    truth = 0 if question['answer'][0]['@correct'] == 'True' else 1
                    examples.append(
                        InputExample(
                            example_id=idx,
                            question=query,
                            contexts=[passage, passage],  # this is not efficient but convenient
                            endings=[options[0], options[1]],
                            label=truth))
                    idx += 1
            except:
                corrupted_sample += 1
                continue
        logger.info(" Corrupted sample :{}".format(corrupted_sample))
        return examples

class DreamProcessor(DataProcessor):
    """Processor for the DREAM 2018 Task 11 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_adv_examples(self, data_dir, adv_type):
        path = os.path.join(data_dir, adv_type+'.pkl')
        with open(path, 'rb') as f:
            examples = torch.load(f)
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    def _create_examples(self, data, type):
        idx = 0
        examples = []
        for item in data:
            passage = ' '.join(item[0])
            for question in item[1]:
                query = question['question']
                options = question['choice']
                truth = options.index(question['answer'])
                examples.append(
                    InputExample(
                        example_id=idx,
                        question=query,
                        contexts=[passage, passage, passage],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2]],
                        label=truth))
                idx += 1
        return examples



def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    no_para=False
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            # if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
            #     logger.info('Attention! you are cropping tokens (swag task is ok). '
            #             'If you are training ARC and RACE and you are poping question + options,'
            #             'you need to try to use a bigger max seq length!')

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            if no_para:
                text_a_np = ' '.join([tokenizer.pad_token]*1)
                text_b_np = ending
                inputs_np = tokenizer.encode_plus(
                    text_a,
                    text_b_np,
                    add_special_tokens=True,
                    max_length=max_length,
                )
                input_ids_np, token_type_ids_np = inputs_np["input_ids"], inputs_np["token_type_ids"]
                attention_mask_np = [1 if mask_padding_with_zero else 0] * len(input_ids_np)
                padding_length_np = max_length - len(input_ids_np)
                if pad_on_left:
                    input_ids_np = ([pad_token] * padding_length_np) + input_ids_np
                    attention_mask_np = ([0 if mask_padding_with_zero else 1] * padding_length_np) + attention_mask_np
                    token_type_ids_np = ([pad_token_segment_id] * padding_length_np) + token_type_ids_np
                else:
                    input_ids_np = input_ids_np + ([pad_token] * padding_length_np)
                    attention_mask_np = attention_mask_np + ([0 if mask_padding_with_zero else 1] * padding_length_np)
                    token_type_ids_np = token_type_ids_np + ([pad_token_segment_id] * padding_length_np)
                assert len(input_ids_np) == max_length
                assert len(attention_mask_np) == max_length
                assert len(token_type_ids_np) == max_length

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids,
                                     input_ids_np, attention_mask_np, token_type_ids_np))


        label = label_map[example.label]
        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]



processors = {
    "race": RaceProcessor,
    "arc": ArcProcessor,
    "mctest": MctestProcessor,
    "semeval2018": Semeval2018Processor,
    "dream": DreamProcessor,
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race": 4,
    "arc": 4,
    "mctest": 4,
    "semeval2018": 2,
    "dream": 3
}
