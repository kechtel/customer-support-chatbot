import argparse
import os
import re

import emoji
import fasttext
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split

model = fasttext.load_model('lid.176.ftz')


def parse_filename(filename):
    _, company, _, tweets = filename.split('.')[0].split('-')
    return company, tweets


def id2text(df, x):
    row = df[df.tweet_id == int(x)]
    # some ids are missing, they just don't exist in data
    return '' if (len(row) == 0) else row.iloc[0]['text']


def clean_tweet(tweet):
    # removes @ mentions, hashtags, emojis, twitter reserved words and numbers
    p.set_options(p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)
    clean = p.clean(tweet)

    # transforms every url to "<url>" token and every hashtag to "<hashtag>" token
    p.set_options(p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER, p.OPT.HASHTAG, p.OPT.URL)
    clean = p.tokenize(clean)
    clean = re.sub(r'\$HASHTAG\$', '<hashtag>', clean)
    clean = re.sub(r'\$URL\$', 'https://t.co/chatbot', clean)

    # preprocessor doesn't seem to clean all emojis so we run text trough emoji regex to clean leftovers
    clean = re.sub(emoji.get_emoji_regexp(), '', clean)

    # removing zero-width character which is often bundled with emojis
    clean = re.sub(u'\ufe0f', '', clean)

    # remove multiple empty spaces with one
    clean = re.sub(r' +', ' ', clean)

    # replace &gt; and &lt;
    clean = re.sub(r'&gt;', '>', clean)
    clean = re.sub(r'&lt;', '<', clean)

    # strip any leftover spaces at the beginning and end
    clean = clean.strip()

    return clean


def set_empty_if_not_english(x):
    return x if model.predict(x.replace('\n', ''))[0][0] == '__label__en' else ''


def qa_from_author(df, author_id):
    """
    Creates qa dataset (in form of dataframe) from all tweets of author (identified by author_id)

    :param df: All twitter customer support data as dataframe.
    :param author_id: Name of author.
    :return: Dataframe containing 'question' and 'answer' fields where 'question' is user tweet and 'answer' is customer
            support tweet
    """
    # get all tweets from certain support service
    support_service = df[df.author_id == author_id]
    # remove tweets which are not triggered by user tweet (there is no Q(uestion))
    support_service = support_service[~support_service.in_response_to_tweet_id.isnull()]

    # take column we are interested in
    support_service = support_service[['author_id', 'text', 'in_response_to_tweet_id']]

    # replace tweet ids with actual tweet text
    support_service.loc[:, 'in_response_to_tweet_id'] = support_service.in_response_to_tweet_id.apply(lambda x: id2text(df, x))

    # rename and rearrange columns
    support_service.rename(columns={'author_id': 'author_id', 'text': 'answer', 'in_response_to_tweet_id': 'question'},
                           inplace=True)
    support_service = support_service[['author_id', 'question', 'answer']]

    # clean twitter data
    support_service.loc[:, 'question'] = support_service.question.apply(clean_tweet)
    support_service.loc[:, 'answer'] = support_service.answer.apply(clean_tweet)

    # filter all languages which are not english (non-english tweets will be set to empty string and then filtered at
    # the end of this method)
    support_service.loc[:, 'question'] = support_service.question.apply(lambda x: set_empty_if_not_english(x))
    support_service.loc[:, 'answer'] = support_service.answer.apply(lambda x: set_empty_if_not_english(x))

    # remove all QA pairs where Q or A are empty or contain only dot (.)
    support_service = support_service[~(support_service.question == '') & ~(support_service.answer == '')]
    support_service = support_service[~(support_service.question == '.') & ~(support_service.answer == '.')]

    return support_service


def split_dataset(path, random_state=287):
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path).split('.')[0]  # file name must end in '.tsv'

    df = pd.read_csv(path, sep='\t')

    train, rest = train_test_split(df, test_size=0.2, random_state=random_state)
    val, test = train_test_split(rest, test_size=0.5, random_state=random_state)

    # write train, val, test
    train.to_csv(dir_name + os.path.sep + file_name + '-train.tsv', sep='\t', index=False)
    val.to_csv(dir_name + os.path.sep + file_name + '-val.tsv', sep='\t', index=False)
    test.to_csv(dir_name + os.path.sep + file_name + '-test.tsv', sep='\t', index=False)


def create_dataset(df, author_ids):
    dataset = qa_from_author(df, author_ids[0])
    for author_id in author_ids[1:]:
        dataset = pd.concat([dataset, qa_from_author(df, author_id)])
    return dataset


def create_and_write_dataset(df, author_id, path):
    """
    Creates tsv dataset which contains only Apple support conversations with customers.
    """
    dataset = create_dataset(df, [author_id])
    dataset_path = path + author_id.lower() + '.tsv'
    dataset.to_csv(dataset_path, sep='\t', index=False)
    split_dataset(dataset_path)


def create_all_dataset(df, path):
    """
    Creates tsv dataset which contains many customer support services from dataset. Included support service authors are
    'AppleSupport', 'AmazonHelp', 'Uber_Support', 'Delta', 'SpotifyCares', 'Tesco', 'AmericanAir',
                  'comcastcares', 'TMobileHelp', 'British_Airways', 'SouthwestAir', 'Ask_Spectrum' and  'hulu_support'
    """
    author_ids = ['AppleSupport', 'AmazonHelp', 'Uber_Support', 'Delta', 'SpotifyCares', 'Tesco', 'AmericanAir',
                  'comcastcares', 'TMobileHelp', 'British_Airways', 'SouthwestAir', 'Ask_Spectrum', 'hulu_support']
    dataset = create_dataset(df, author_ids)
    dataset = dataset.sample(frac=1)  # shuffle dataset
    dataset_path = path + 'twitter-all' + '.tsv'
    dataset.to_csv(dataset_path, sep='\t', index=False)
    split_dataset(dataset_path)


def main():
    parser = argparse.ArgumentParser(description='Script for formatting training data for seq2seq chatbot.')
    parser.add_argument('--data-path', help='Path where to find training data.')
    args = parser.parse_args()

    if not os.path.exists('data'):
        os.mkdir('data')

    for file in os.listdir(args.data_path):
        df = pd.read_excel(os.path.join(args.data_path, file))
        company, _ = parse_filename(file)
        create_and_write_dataset(df, company, 'data/')


if __name__ == '__main__':
    main()
