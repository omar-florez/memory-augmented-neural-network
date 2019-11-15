import pandas as pd
import os
import json
import ipdb
import numpy as np
import csv

PATH = '/Users/ost437/Documents/OneDrive/workspace/datasets/kvret_dataset_public'

def run():
    file_names = ['kvret_train_public.json', 'kvret_test_public.json', 'kvret_dev_public.json']
    entity_names = 'kvret_entities.json'

    num_dialogues = 0
    num_turns = 0

    output = []
    for fn in file_names:
        with open(os.path.join(PATH, fn)) as f:
            data = json.load(f)
            num_dialogues += len(data)
            num_turns += sum([len(data[i]['dialogue']) for i in range(len(data))])

            # iterate on each dialogue
            for dialogue_id in range(len(data)):
                # iterate on each turn of this dialogue
                for turn_id in range(0, len(data[dialogue_id]['dialogue']), 2):
                    if turn_id+1 < len(data[dialogue_id]['dialogue']) and \
                        'slots' in data[dialogue_id]['dialogue'][turn_id + 1]['data'].keys() and \
                            not data[dialogue_id]['dialogue'][turn_id + 1]['data']['end_dialogue']:
                        utterance = data[dialogue_id]['dialogue'][turn_id]['data']['utterance'].lower()
                        slots = list(data[dialogue_id]['dialogue'][turn_id + 1]['data']['slots'].keys())
                        values = list(data[dialogue_id]['dialogue'][turn_id + 1]['data']['slots'].values())
                        if len(slots)>0:
                            driver_turn = {'utterance': utterance, 'slots': slots, 'values': values}
                            output.append(driver_turn)

    # write raw in canonized form
    os.makedirs(os.path.join('data', 'dialog_state_smtd', 'raw'), exist_ok=True)
    os.makedirs(os.path.join('data', 'dialog_state_smtd', 'tokens'), exist_ok=True)
    os.makedirs(os.path.join('data', 'dialog_state_smtd', 'lm'), exist_ok=True)
    os.makedirs(os.path.join('data', 'dialog_state_smtd', 'results'), exist_ok=True)

    dataset = []
    for entry in output:
        x = entry['utterance']
        y = '&'.join(np.sort(entry['slots']))
        dataset.append({'x': x, 'y': y})
    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(os.path.join('data', 'dialog_state_smtd', 'raw', 'dataset.tsv'), sep='\t')

    # write list of unique labels
    labels = np.unique(dataset_df['y'])
    label_counts = dataset_df.groupby(['y']).count()
    label_counts.to_csv(os.path.join('data', 'dialog_state_smtd', 'tokens', 'label_counts.tsv'), sep='\t')


    wtr = csv.writer(open(os.path.join('data', 'dialog_state_smtd', 'tokens', 'label_map.txt'), 'w'),
                     delimiter=',', lineterminator='\n')
    valid_labels_counts = label_counts[label_counts['x'] > 10]
    labels = valid_labels_counts.index.values
    for label in labels:
        wtr.writerow([label])

    # write train, validation, and test datasets
    dataset_df = dataset_df[dataset_df['y'].isin(labels)]
    index = np.arange(len(dataset_df))
    np.random.shuffle(index)
    train_dataset = dataset[0 : int(len(index)*0.8)]
    valid_dataset = dataset[int(len(index)*0.8) : len(index)]
    train_df = pd.DataFrame(train_dataset)
    valid_df = pd.DataFrame(valid_dataset)
    train_df.to_csv(os.path.join('data', 'dialog_state_smtd', 'tokens', 'train.tsv'), sep='\t')
    valid_df.to_csv(os.path.join('data', 'dialog_state_smtd', 'tokens', 'valid.tsv'), sep='\t')
    avg_sequence_len = np.mean([len(xx) for xx in dataset_df['x']])

    print('Number of dialogues: {}'.format(num_dialogues))
    print('Number of turns: {}'.format(num_turns))
    print('Number of observations (valid turns): {}'.format(len(dataset)))
    print('Number of observations (valid turns after removing low frequent labels): {}'.format(len(dataset_df)))
    print('Number of training observations (valid turns): {}'.format(len(train_df)))
    print('Number of validation observations (valid turns): {}'.format(len(valid_df)))
    print('Number of labels: {}'.format(len(labels)))
    print('Average seuence length: {}'.format(avg_sequence_len))



    ipdb.set_trace()



if __name__ == '__main__':
    run()
