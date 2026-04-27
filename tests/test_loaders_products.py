"""Tests for products config loader (v2.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

from configs.loaders import get_product, load_products_config


@pytest.fixture
def products_yaml(tmp_path):
    f = tmp_path / "products.yaml"
    f.write_text(yaml.dump({
        "default_product_id": "TEST",
        "products": {
            "TEST": {
                "name": "test product",
                "bottle_volume_ml": 500,
                "flow_regime": "laminar",
                "l_ref_mm": 0.5,
                "q_threshold": 1.0e-3,
            },
            "P001": {
                "name": "500mL water",
                "bottle_volume_ml": 500,
                "flow_regime": "laminar",
                "l_ref_mm": 0.5,
                "q_threshold": 2.0e-3,
            },
        },
    }, allow_unicode=True), encoding="utf-8")
    return f


class TestLoadProductsConfig:
    def test_load_basic(self, products_yaml):
        cfg = load_products_config(products_yaml)
        assert cfg["default_product_id"] == "TEST"
        assert "P001" in cfg["products"]

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_products_config(tmp_path / "missing.yaml")

    def test_load_real_repo_config(self):
        cfg = load_products_config()  # default = configs/products.yaml
        assert "products" in cfg
        # Repo ships with TEST + P001 + P002 + P003 minimum
        for pid in ("TEST", "P001", "P002", "P003"):
            assert pid in cfg["products"]


class TestGetProduct:
    def test_known_product(self, products_yaml):
        cfg = load_products_config(products_yaml)
        p = get_product(cfg, "P001")
        assert p["name"] == "500mL water"
        assert p["q_threshold"] == pytest.approx(2.0e-3)

    def test_unknown_falls_back_to_default(self, products_yaml):
        cfg = load_products_config(products_yaml)
        p = get_product(cfg, "DOES_NOT_EXIST")
        # Fallback to default_product_id = "TEST"
        assert p["name"] == "test product"

    def test_no_default_returns_empty_dict(self, tmp_path):
        f = tmp_path / "minimal.yaml"
        f.write_text(yaml.dump({"products": {"P001": {"q_threshold": 1e-3}}}),
                     encoding="utf-8")
        cfg = load_products_config(f)
        # No default_product_id → unknown lookup gets empty dict, not None
        p = get_product(cfg, "DOES_NOT_EXIST")
        assert p == {}

    def test_real_test_product_has_q_threshold(self):
        cfg = load_products_config()
        p = get_product(cfg, "TEST")
        assert "q_threshold" in p
        assert p["q_threshold"] > 0
