### Script to evaluate the final seq2seq model
from evaluation_utils import *
from training import *
import hparams

import datetime
import time

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Self-supervised Style Transfer LM")

parser.add_argument('--device',
                    default='cuda',
                    type=str,
                    help="Set the device for the model to run on, with 'cuda' or 'cpu'")

parser.add_argument('--dataset',
                    default='yelp_20',
                    type=str,
                    help='Specify the dataset to use: yelp_#, imdb_# or multi_# with # replacec by the % noised')

parser.add_argument('--translation_model_path',
                    default='./saved_models/yelp_translation_model.pt',
                    type=str,
                    help="Set the path and filename to load a pre-trained language model")

parser.add_argument('--test_path',
                    default='./data/yelp_test_20.csv',
                    type=str,
                    help='Specify the path/file_name.csv for the testing data')

parser.add_argument('--classifier_path',
                    default='',
                    type=str,
                    help="Set the path and filename to load a pre-trained classifer")

parser.add_argument('--style_tokens',
                    default=['<pos>', '<neg>'],
                    type=list,
                    help='Specify the tokens the model is using as style tokens')

parser.add_argument('--model_type',
                    default='BART',
                    type=str,
                    help='Specify model, either BART or DeleteAndRetrieve')

args = parser.parse_args()

def main():
    tokenizer =  BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    special_tokens_dict = {'additional_special_tokens': ['<pos>', '<neg>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    if args.model_type == 'BART':
        lm_model = torch.load(args.translation_model_path)
        print(f'--- Running Validation Check on {args.dataset} - {args.model_type} model ---')
        val_data = auto_evaluation_seq2seq(lm_model=lm_model, 
                                        tokenizer=tokenizer,  
                                        classifier_path=args.classifier_path, 
                                        val_path=args.test_path,
                                        num_samples=None,
                                        translator_training=True,
                                        shuffle=False)
        val_acc = val_data['val_acc']
        val_examples = pd.DataFrame({'Original': val_data['val_source'],
                                    'Translated': val_data['generated_sentences'],
                                    'Desired Label': val_data['labels'] })
        # Write example translated sentences to file
        if not os.path.exists('./reports'):
            os.makedirs('./reports')
        val_examples.to_csv(f'./reports/{args.dataset}_val_data_{np.round(val_acc,1)}_final.csv', index=False)
    else:
        # The DeleteAndRetrieve model generates validation examples after each epoch, so we can just use
        # the latest "preds" file. The model translates to "positive", so we just need to check that everything
        # it translated was positive
        print(f'--- Running Validation Check on {args.dataset} - {args.model_type} model ---')
        assert len(args.test_path) > 0, "You must enter a test path for the data"
        data_path = args.test_path.split('.')
        data_path_pos = data_path[0] + '_1.' + data_path[1]
        data_path_neg = data_path[0] + '_0.' + data_path[1]
        val_data = auto_evaluation_seq2seq(lm_model=None, 
                                        tokenizer=tokenizer,  
                                        classifier_path=args.classifier_path, 
                                        val_path=None,
                                        num_samples=None,
                                        val_sentences=data_path_pos,
                                        drg_label=1)
        val_acc = val_data['val_acc']

        val_examples = pd.DataFrame({'Original': val_data['val_source'],
                            'Translated': val_data['generated_sentences'],
                            'Desired Label': val_data['labels'] })

        val_data = auto_evaluation_seq2seq(lm_model=None, 
                                        tokenizer=tokenizer,  
                                        classifier_path=args.classifier_path, 
                                        val_path=None,
                                        num_samples=None,
                                        val_sentences=data_path_neg,
                                        drg_label=0)
        val_acc += val_data['val_acc']
        val_acc = (val_acc / 2) # This should be fine as there are the same number of egs in each set

        val_examples = pd.concat([val_examples,
                                  pd.DataFrame({'Original': val_data['val_source'],
                                  'Translated': val_data['generated_sentences'],
                                   'Desired Label': val_data['labels'] })])


        # Write example translated sentences to file
        if not os.path.exists('./reports'):
            os.makedirs('./reports')
        val_examples.to_csv(f'./reports/{args.dataset}_drg_val_data_{np.round(val_acc,1)}_final.csv', index=False)

    # Save the file
    if not os.path.exists('./reports'):
        os.makedirs('./reports')
    with open(f'./reports/val_acc_{args.dataset}_{datetime.datetime.now().strftime("%d-%m-%Y-%H")}.txt', 'w') as f:
        f.write(f'Accuracy: {val_acc} \n')
        f.write(f'Dataset: {args.dataset} \n')
        f.write(f'Model: {args.model_type}')

if __name__ == "__main__":
    main()