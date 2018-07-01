from helpers import *
import numpy

class TestHelpersMod(object):
    @classmethod
    def setup_class(self):
        self.TRAIN_CSV = 'data/external/train.csv'
        self.TRAIN_FILES = 'data/external/audio_train'

    def test_find_paths_with_tags(self):
        multiple_filters = find_paths_with_tags(csv_path=self.TRAIN_CSV,
                                                files_path=self.TRAIN_FILES,
                                                filters=['Harmonica', 'Fireworks'])

        assert len(multiple_filters) == 465
        assert 'data/external/audio_train/01302128.wav' in [ x['path'] for x in multiple_filters ]
        assert 'data/external/audio_train/01811e48.wav' in [ x['path'] for x in multiple_filters ]


        single_filter = find_paths_with_tags(self.TRAIN_CSV,
                                             self.TRAIN_FILES,
                                             ['Fart'])
        assert len(single_filter) == 300
        assert 'data/external/audio_train/00c82919.wav' in [ x['path'] for x in single_filter ]

        no_filter = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES)
        assert len(no_filter) == 9473

        with_limit = find_paths_with_tags(self.TRAIN_CSV,
                                          self.TRAIN_FILES,
                                          ['Fireworks'],
                                          limit=1)

        assert len(with_limit) == 1

    def test_load_wav_files(self):
        all_farts = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES,
                                         ['Fart'],
                                         limit=2)

        train_wav_inputs = load_wav_files(all_farts)
        assert len(train_wav_inputs) == 2

        # assert wav base 0 is n dimensional array
        sample = train_wav_inputs[0]
        assert isinstance(sample.wav[0], numpy.ndarray)

    def test_normalize_audio(self):
        all_cellos = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES,
                                         ['Cello'],
                                         limit=2)

        train_wav_inputs = load_wav_files(all_cellos)

        normalizer = NormalizeAudio(train_wav_inputs)
        train_wav_inputs_1s = normalizer.sample_rate(22050)
        assert [x.wav.shape[0] for x in train_wav_inputs_1s] == [22050, 22050]
