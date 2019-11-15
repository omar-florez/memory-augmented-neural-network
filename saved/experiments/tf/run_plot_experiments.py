from utils import visualization_experiments as viz
import os


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root, 'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']

            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
            )

            datasets.append(experiment_data)
            unit += 1

    return datasets


if __name__ == '__main__':
    root_folder = '/Users/ost437/Documents/OneDrive/workspace/WorkspaceCapitalOne/learning_to_hash_labels/saved/experiments/tf/precision_validation'
    filepaths = ['run_summary_lth_20_2.19@lstm, key_dim=256, seq_len_words=20, keep_prob=1.0, num_layers_encoder=3, input_encoding=word_random, dataset=dialogue_states-tag-Precision_precision_validation.csv',
                 'run_summary_lth_20_2.20@lth+mem_size=8000, key_dim=256, seq_len_words=20, keep_prob=1.0, num_layers_encoder=3, input_encoding=word_random, dataset=dialogue_states, memories=lth-tag-Precision_precision_validation.csv',
                 'run_summary_lth_20_2.20@lth_gmm+mem_size=8000, key_dim=256, seq_len_words=20, keep_prob=1.0, num_layers_encoder=3, input_encoding=word_random, dataset=dialogue_states, memories=lth_gmm-tag-Precision_precision_validation.csv',
                 'run_summary_lth_20_2.22@lth_gmm+mem_size=8000, key_dim=256, seq_len_words=20, keep_prob-0.95, num_layers_encoder=3, input_encoding=word_random, dataset=dialogue_states, memories=lth_gmm-tag-Precision_precision_validation.csv',
                 'run_summary_lth_20_2.28@lth_gmm+mem_size=8000, key_dim=256, seq_len_words=20, keep_prob-0.7, num_layers_encoder=3, input_encoding=word_random, dataset=dialogue_states, memories=lth_gmm-tag-Precision_precision_validation.csv']

    labels = ['a', 'b', 'c', 'd', 'e']

    filepaths = [os.path.join(root_folder, f) for f in filepaths]
    output_folder = './'
    output_filepath = os.path.join(output_folder, 'output_tensorflow_experiment.png')
    viz.plot_tf_csv_files_from_folder(filepaths=filepaths, labels=labels, output_filepath=output_filepath)

