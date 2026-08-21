import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_jlens_range.py"
    spec = importlib.util.spec_from_file_location("run_jlens_range", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_combined_catalog_has_stable_domain_order_and_one_based_indexes():
    module = _load_module()
    task_sets = {
        "airline": [SimpleNamespace(id="a0"), SimpleNamespace(id="a1")],
        "retail": [SimpleNamespace(id="r0")],
        "telecom": [SimpleNamespace(id="t0"), SimpleNamespace(id="t1")],
    }

    catalog = module.build_catalog(task_sets)

    assert [entry.index for entry in catalog] == [1, 2, 3, 4, 5]
    assert [entry.domain for entry in catalog] == [
        "airline",
        "airline",
        "retail",
        "telecom",
        "telecom",
    ]


def test_global_range_can_cross_domain_boundaries():
    module = _load_module()
    task_sets = {
        "airline": [SimpleNamespace(id="a0"), SimpleNamespace(id="a1")],
        "retail": [SimpleNamespace(id="r0"), SimpleNamespace(id="r1")],
        "telecom": [SimpleNamespace(id="t0")],
    }
    catalog = module.build_catalog(task_sets)

    selected = module.select_range(catalog, start=2, count=3)

    assert [(entry.index, entry.domain) for entry in selected] == [
        (2, "airline"),
        (3, "retail"),
        (4, "retail"),
    ]


@pytest.mark.parametrize(
    ("start", "count"),
    [(0, 1), (1, 0), (6, 1), (4, 3)],
)
def test_global_range_rejects_invalid_or_partial_ranges(start, count):
    module = _load_module()
    task_sets = {
        "airline": [SimpleNamespace(id="a0")],
        "retail": [SimpleNamespace(id="r0")],
        "telecom": [SimpleNamespace(id="t0"), SimpleNamespace(id="t1")],
    }
    catalog = module.build_catalog(task_sets)

    with pytest.raises(ValueError):
        module.select_range(catalog, start=start, count=count)


def test_run_config_uses_gpt_oss_trajectory_path_and_full_task_set(tmp_path):
    module = _load_module()
    args = SimpleNamespace(
        profile="qwen3.5-4b",
        trajectory_root=tmp_path,
        agent_api_base="http://127.0.0.1:8000/v1",
        temperature=1.0,
        max_tokens=4096,
        user_model="gpt-5.2-2025-12-11",
        user_api_base="https://api.openai.com/v1",
        num_trials=1,
        max_concurrency=4,
        seed=300,
        auto_resume=False,
    )

    config = module._build_config(
        args,
        domain="telecom",
        task_ids=["persona-expanded-task"],
        run_label="0165-0165",
    )

    assert config.task_split_name is None
    assert config.task_ids == ["persona-expanded-task"]
    assert config.agent == "llm_agent"
    assert config.llm_agent == "hosted_vllm/Qwen3.5-4B"
    assert config.llm_args_agent == {
        "api_base": "http://127.0.0.1:8000/v1",
        "temperature": 1.0,
        "max_tokens": 4096,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    assert config.llm_user == "gpt-5.2-2025-12-11"
    assert config.llm_args_user == {
        "api_base": "https://api.openai.com/v1",
        "reasoning_effort": "low",
    }
    assert config.max_concurrency == 4
    assert config.seed == 300
    assert config.verbose_logs is True
    assert config.save_to == str((tmp_path / "0165-0165" / "telecom").resolve())


def test_generation_defaults_match_gpt_oss_benchmark():
    module = _load_module()

    args = module.parse_args([])

    assert args.profile == "qwen3.5-4b"
    assert args.user_model == "gpt-5.2-2025-12-11"
    assert args.temperature == 1.0
    assert args.max_tokens == 4096
    assert args.max_concurrency == 4
    assert args.seed == 300
