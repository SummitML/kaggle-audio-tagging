from src.util import *

class TestUtilMod(object):
    def test_find_paths_with_tags(self):
        multiple_filters = find_paths_with_tags('data/external/train.csv', ['Harmonica', 'Fireworks'])
        assert len(multiple_filters) == 465
        assert '16ef3c60.wav' in multiple_filters
        assert '18b3013e.wav' in multiple_filters


        single_filter = find_paths_with_tags('data/external/train.csv', ['Fart'])
        assert len(single_filter) == 300
        assert '16df86bb.wav' in single_filter

        no_filter = find_paths_with_tags('data/external/train.csv')
        assert len(no_filter) == 9473
