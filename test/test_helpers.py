from helpers import *

class TestHelpersMod(object):
    def test_find_paths_with_tags(self):
        multiple_filters = find_paths_with_tags(csv_path='data/external/train.csv',
                                                files_path='data/external/audio_train',
                                                filters=['Harmonica', 'Fireworks'])

        assert len(multiple_filters) == 465
        assert 'data/external/audio_train/01302128.wav' in [ x['path'] for x in multiple_filters ]
        assert 'data/external/audio_train/01811e48.wav' in [ x['path'] for x in multiple_filters ]


        single_filter = find_paths_with_tags('data/external/train.csv',
                                             'data/external/audio_train',
                                             ['Fart'])
        assert len(single_filter) == 300
        assert 'data/external/audio_train/00c82919.wav' in [ x['path'] for x in single_filter ]

        no_filter = find_paths_with_tags('data/external/train.csv',
                                         'data/external/audio_train')
        assert len(no_filter) == 9473

        with_limit = find_paths_with_tags('data/external/train.csv',
                                     'data/external/audio_train',
                                     ['Fireworks'],
                                     limit=1)

        assert len(with_limit) == 1

    def test_load_wav_files(self):
        all_farts = find_paths_with_tags('data/external/train.csv',
                                     'data/external/audio_train',
                                     ['Fart'],
                                     limit=2)

        train_wav_inputs = load_wav_files(all_farts)
        assert len(train_wav_inputs) == 2
