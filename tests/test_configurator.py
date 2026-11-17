import pytest

from configurator import load_config

DEFAULTS = dict(batch_size=12, learning_rate=6e-4, compile=True, dataset='openwebtext',
                run_name='run', vocab_size=None)


def test_defaults_are_copied():
    config = load_config(DEFAULTS, [])
    assert config == DEFAULTS and config is not DEFAULTS


def test_cli_overrides():
    config = load_config(DEFAULTS, ['--batch_size=32', '--learning_rate=1e-3', '--compile=False',
                                    '--dataset=fineweb', '--vocab_size=32000'])
    assert config['batch_size'] == 32
    assert config['learning_rate'] == 1e-3
    assert config['compile'] is False
    assert config['dataset'] == 'fineweb'
    assert config['vocab_size'] == 32000


def test_int_is_promoted_to_float_and_strings_stay_strings():
    config = load_config(DEFAULTS, ['--learning_rate=1', '--run_name=123', "--dataset='c4'"])
    assert config['learning_rate'] == 1.0 and isinstance(config['learning_rate'], float)
    assert config['run_name'] == '123'
    assert config['dataset'] == 'c4'


@pytest.mark.parametrize('arg', ['--batch_size=1.5', '--batch_size=True', '--compile=1',
                                 '--learning_rate=fast', '--batch_size=None', '--unknown=1',
                                 '--batch_size'])
def test_invalid_overrides_are_rejected(arg):
    with pytest.raises(ValueError):
        load_config(DEFAULTS, [arg])


def test_config_files_and_precedence(tmp_path):
    first = tmp_path / 'first.py'
    first.write_text("import math\n_helper = 2\nbatch_size = 4 * _helper\ndataset = 'c4'\n")
    second = tmp_path / 'second.py'
    second.write_text("dataset = 'fineweb'\n")
    config = load_config(DEFAULTS, [str(first), str(second), '--batch_size=64'])
    assert config['batch_size'] == 64  # later arguments win
    assert config['dataset'] == 'fineweb'
    config = load_config(DEFAULTS, ['--batch_size=64', str(first)])
    assert config['batch_size'] == 8


def test_config_file_with_unknown_or_mistyped_key(tmp_path):
    bad = tmp_path / 'bad.py'
    bad.write_text("learning_rat = 1e-3\n")
    with pytest.raises(ValueError, match='learning_rat'):
        load_config(DEFAULTS, [str(bad)])
    bad.write_text("batch_size = '12'\n")
    with pytest.raises(ValueError, match='batch_size'):
        load_config(DEFAULTS, [str(bad)])
