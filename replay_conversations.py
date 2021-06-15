import argparse
import os

import pandas as pd
import torch

from dataset import field_factory, metadata_factory
from model import predict_model_factory
from predict import ModelDecorator, get_model_path
from serialization import load_object


def load_model(model_path, epoch):
    torch.set_grad_enabled(False)

    model_args = load_object(os.path.join(model_path, 'args'))
    print('Model args loaded.')
    vocab = load_object(os.path.join(model_path, 'vocab'))
    print('Vocab loaded.')

    cuda = torch.cuda.is_available()
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    print("Using %s for inference" % ('GPU' if cuda else 'CPU'))

    field = field_factory(model_args)
    field.vocab = vocab
    metadata = metadata_factory(model_args, vocab)

    model = ModelDecorator(
        predict_model_factory(model_args, metadata, get_model_path(model_path + os.path.sep, epoch), field))
    print('model loaded')
    model.eval()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for replaying conversations with a trained seq2seq chatbot.')
    parser.add_argument('--filename', help='File of conversations to replay.')
    parser.add_argument('-p', '--model-path',
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    args = parser.parse_args()

    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists(os.path.join('data', 'replayed')):
        os.mkdir(os.path.join('data', 'replayed'))

    model = load_model(args.model_path, args.epoch)

    df = pd.read_excel(args.filename)
    df['response'] = df.apply(
        lambda x: model(x['text'], sampling_strategy='greedy', max_seq_len=50) if x['inbound'] else '', axis=1)

    df['text'] = df.apply(lambda x: '' if not x['inbound'] else x['text'], axis=1)

    df['text'] = df.apply(
        lambda x: df[(df['main_tweet_id'] == x['main_tweet_id'])
                     & (df['created_at'] < x['created_at'])
                     & (df['inbound'])].sort_values(by='created_at', ascending=False)['response'].values[0]
        if len(df[(df['main_tweet_id'] == x['main_tweet_id']) & (df['created_at'] < x['created_at'])
                  & (df['inbound'])].sort_values(by='created_at', ascending=False)['response'].values) > 0
           and not x['inbound'] else x['text'], axis=1)

    df = df.drop('response', axis=1)

    df.to_excel(os.path.join('data', 'replayed', args.filename.split('/')[-1].replace('.xlsx', '') + '-replayed.xlsx'))
