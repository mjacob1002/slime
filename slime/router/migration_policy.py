"""Request migration policies for StreamingRouter.

Determines whether and how to migrate truncated requests to a different engine.
"""

from slime.utils.types import Sample


class RequestMigrationPolicy:
    """Base class for request migration policies."""

    def should_migrate(self, sample: Sample, engine_rank: int) -> bool:
        raise NotImplementedError

    def get_migration_target(self, sample: Sample, engine_rank: int) -> int:
        raise NotImplementedError

    def adjust_sampling_params(self, sample: Sample, sampling_params: dict) -> dict:
        raise NotImplementedError


class NoMigrationPolicy(RequestMigrationPolicy):
    """No migration — truncated samples stay as-is."""

    def should_migrate(self, sample: Sample, engine_rank: int) -> bool:
        return False

    def get_migration_target(self, sample: Sample, engine_rank: int) -> int:
        return engine_rank

    def adjust_sampling_params(self, sample: Sample, sampling_params: dict) -> dict:
        return sampling_params
