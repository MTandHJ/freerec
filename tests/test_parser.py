from freerec.dict2obj import Config
from freerec.parser import CONFIG, CORE_CONFIG, CoreParser


class TestCONFIG:
    """Test the default CONFIG and CORE_CONFIG objects."""

    def test_config_has_required_keys(self):
        assert CONFIG.SAVED_FILENAME == "model.pt"
        assert CONFIG.BEST_FILENAME == "best.pt"
        assert CONFIG.CHECKPOINT_FREQ == 1
        assert "model" in CONFIG.CHECKPOINT_MODULES
        assert CONFIG.SUMMARY_FILENAME == "SUMMARY.md"
        assert CONFIG.which4best == "LOSS"

    def test_config_is_config_instance(self):
        assert isinstance(CONFIG, Config)

    def test_core_config_has_required_keys(self):
        assert CORE_CONFIG.EXCLUSIVE is False
        assert CORE_CONFIG.COMMAND is None
        assert isinstance(CORE_CONFIG.ENVS, dict)
        assert isinstance(CORE_CONFIG.PARAMS, dict)
        assert isinstance(CORE_CONFIG.DEFAULTS, dict)
        assert CORE_CONFIG.log2file is True
        assert CORE_CONFIG.log2console is True

    def test_core_config_inherits_monitor_best(self):
        assert CORE_CONFIG.MONITOR_BEST_FILENAME == CONFIG.MONITOR_BEST_FILENAME


class TestCoreParser:
    """Test CoreParser initialization and configuration."""

    def test_init(self):
        parser = CoreParser()
        assert isinstance(parser, Config)
        assert parser.EXCLUSIVE is False
        assert parser.COMMAND is None
