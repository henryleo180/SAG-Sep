from __future__ import annotations

from sagsep_train.data.format import (
    FEATURE_CONTRACT,
    SHARD_AXES,
    SHARD_DTYPES,
    SHARD_KEYS,
    SHARD_SCHEMA_VERSION,
    SUPPORT_RULE,
    TERMINATION_CODES,
    TERMINATION_NAMES,
    InstanceRecord,
    ShardBuilder,
    ShardData,
    StateRecord,
    StateView,
    ceil_div,
    is_sha256_hex,
    merge_shards,
    read_shard,
    read_shard_counts,
    sha256_file,
    write_shard,
)
from sagsep_train.data.manifest import (
    CREATED_BY_PACKAGE,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    SHARD_GLOB,
    SPLITS,
    DatasetManifest,
    ShardEntry,
    build_manifest,
    read_manifest,
    write_manifest,
)
from sagsep_train.data.validate import (
    ValidationError,
    assert_valid_dataset,
    assert_valid_shard,
    validate_dataset,
    validate_shard,
)

__all__ = [
    "SHARD_SCHEMA_VERSION",
    "FEATURE_CONTRACT",
    "SUPPORT_RULE",
    "TERMINATION_CODES",
    "TERMINATION_NAMES",
    "SHARD_KEYS",
    "SHARD_DTYPES",
    "SHARD_AXES",
    "InstanceRecord",
    "StateRecord",
    "StateView",
    "ShardBuilder",
    "ShardData",
    "write_shard",
    "read_shard",
    "read_shard_counts",
    "merge_shards",
    "sha256_file",
    "is_sha256_hex",
    "ceil_div",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "CREATED_BY_PACKAGE",
    "SHARD_GLOB",
    "SPLITS",
    "ShardEntry",
    "DatasetManifest",
    "write_manifest",
    "read_manifest",
    "build_manifest",
    "ValidationError",
    "validate_shard",
    "assert_valid_shard",
    "validate_dataset",
    "assert_valid_dataset",
]

# --------------------------------------------------------------------------
# Training-time surface (torch)
# --------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, str] = {
    "SHARD_SUFFIX": "sagsep_train.data.dataset",
    "shard_file_name": "sagsep_train.data.dataset",
    "ShardIndex": "sagsep_train.data.dataset",
    "detach_state": "sagsep_train.data.dataset",
    "TrainingDataset": "sagsep_train.data.dataset",
    "GlobalPermutationSampler": "sagsep_train.data.dataset",
    "ShardOrderSampler": "sagsep_train.data.dataset",
    "select_val_records": "sagsep_train.data.dataset",
    "write_record_ids": "sagsep_train.data.dataset",
    "read_record_ids": "sagsep_train.data.dataset",
    "make_dataloader": "sagsep_train.data.dataset",
    "worker_init_fn_for": "sagsep_train.data.dataset",
    "SUPPORT_RULE_THRESHOLDS": "sagsep_train.data.collate",
    "TrainBatch": "sagsep_train.data.collate",
    "state_instance": "sagsep_train.data.collate",
    "state_features": "sagsep_train.data.collate",
    "collate_states": "sagsep_train.data.collate",
    "collator": "sagsep_train.data.collate",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ += list(_LAZY_EXPORTS)
