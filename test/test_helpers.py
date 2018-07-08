from helpers import *
import numpy as np
import os
import shutil

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
        assert isinstance(sample.wav[0], np.ndarray)

    def test_serialize_deserialize_wavs(self):
        all_cellos = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES,
                                         ['Cello'],
                                         limit=2)

        train_wav_inputs = load_wav_files(all_cellos)
        train_wav_inputs = [ x.__dict__ for x in train_wav_inputs ]

        serialized = serialize_wavs(train_wav_inputs)
        wav = serialized[0]
        assert  isinstance(wav, bytes)

        deserialized = deserialize_wavs(serialized)
        wav, _ = deserialized[0].get('wav')
        assert isinstance(wav, np.ndarray)

    def test_wavs_io(self):
        all_cellos = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES,
                                         ['Cello'],
                                         limit=2)

        train_wav_inputs = load_wav_files(all_cellos)
        train_wav_inputs = [ x.__dict__ for x in train_wav_inputs ]
        out_file = 'tmp/__test__.pkl'

        # test file out
        serialized = serialize_wavs(train_wav_inputs)
        pickle_out(out_file, serialized)

        with open('tmp/__test__.pkl', 'rb') as f:
            pickle.load(f)

        batched_wavs = assemble_batched_wavs('tmp')
        assert len(batched_wavs) == 1

        shutil.rmtree('tmp')

    def test_normalize_audio(self):
        all_cellos = find_paths_with_tags(self.TRAIN_CSV,
                                         self.TRAIN_FILES,
                                         ['Snare_drum'],
                                         limit=2)

        train_wav_inputs = load_wav_files(all_cellos)

        normalizer = NormalizeAudio(train_wav_inputs)
        train_wav_inputs_1s = normalizer.sample_rate(22050)
        assert isinstance(train_wav_inputs_1s[0].wav, tuple)
        assert isinstance(train_wav_inputs_1s[0].wav[0], np.ndarray)
        assert [x.wav[0].shape[0] for x in train_wav_inputs_1s] == [22050, 22050]
