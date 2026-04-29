import dataclasses
import inspect
import json
import os
import sys
from typing import Annotated

import typer


def dataclass_cli_with_json(func, env_var_prefix: str = "SLIME_SCRIPT_"):
    """Drop-in for ``dataclass_cli`` that also accepts ``--config-path PATH``.

    Loads the JSON file at PATH and seeds ``{env_var_prefix}<KEY>`` env vars
    from its keys before delegating to ``dataclass_cli``. Typer's existing
    ``envvar=`` handling then picks them up as defaults.

    Precedence: CLI flag > pre-set env var > JSON value > dataclass default.

    Behavior:
    - Unknown JSON keys are seeded into env vars regardless; if no field
      matches, typer simply ignores the env var. A warning is printed for
      visibility on typos.
    - JSON ``null`` is skipped so the dataclass default applies.
    - Nested values (dict/list) are skipped with a warning — the current
      dataclasses use only scalar fields.
    - Bools are serialized as "true"/"false" so typer parses them correctly.
    - If multiple ``--config-path`` flags are present, the last one wins.
    - The selected path is stashed in ``SLIME_LAUNCHER_CONFIG_PATH`` so
      execute_train can record it in the per-run config snapshot.
    """
    args = sys.argv[1:]
    config_path: str | None = None
    new_argv: list[str] = []
    cli_keys: set[str] = set()
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--config-path":
            if i + 1 >= len(args):
                raise SystemExit("error: --config-path requires a value")
            config_path = args[i + 1]
            i += 2
            continue
        if a.startswith("--config-path="):
            config_path = a.split("=", 1)[1]
            i += 1
            continue
        # Track which fields the CLI explicitly set so JSON values never override them.
        if a.startswith("--no-"):
            cli_keys.add(a[len("--no-"):].split("=", 1)[0].replace("-", "_").lower())
        elif a.startswith("--"):
            cli_keys.add(a.lstrip("-").split("=", 1)[0].replace("-", "_").lower())
        new_argv.append(a)
        i += 1

    if config_path is not None:
        with open(config_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise SystemExit(
                f"--config-path: expected a JSON object at top level, got {type(data).__name__}"
            )
        # Introspect the wrapped function's first arg to find the dataclass —
        # used to warn on JSON keys that don't map to any field (catches typos).
        try:
            param = next(iter(inspect.signature(func).parameters.values()))
            known_fields = {f.name for f in dataclasses.fields(param.annotation)}
        except (StopIteration, TypeError):
            known_fields = None
        os.environ["SLIME_LAUNCHER_CONFIG_PATH"] = str(config_path)
        for k, v in data.items():
            if k.lower() in cli_keys:
                continue
            if isinstance(v, (dict, list)):
                print(
                    f"[dataclass_cli_with_json] WARNING: skipping nested key '{k}' "
                    "(only scalar JSON values are supported)",
                    file=sys.stderr,
                )
                continue
            if v is None:
                continue
            if known_fields is not None and k not in known_fields:
                print(
                    f"[dataclass_cli_with_json] WARNING: JSON key '{k}' has no matching "
                    f"dataclass field on {param.annotation.__name__}; ignoring",
                    file=sys.stderr,
                )
                continue
            env_key = f"{env_var_prefix}{k.upper()}"
            if env_key in os.environ:
                continue
            os.environ[env_key] = "true" if v is True else "false" if v is False else str(v)
        sys.argv = [sys.argv[0]] + new_argv

    return dataclass_cli(func, env_var_prefix=env_var_prefix)


def dataclass_cli(func, env_var_prefix: str = "SLIME_SCRIPT_"):
    """Modified from https://github.com/fastapi/typer/issues/154#issuecomment-1544876144"""

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    dataclass_cls = param.annotation
    assert dataclasses.is_dataclass(dataclass_cls)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(dataclass_cls.__init__)
    old_parameters = list(signature.parameters.values())
    if len(old_parameters) > 0 and old_parameters[0].name == "self":
        del old_parameters[0]

    new_parameters = []
    for param in old_parameters:
        env_var_name = f"{env_var_prefix}{param.name.upper()}"
        new_annotation = Annotated[param.annotation, typer.Option(envvar=env_var_name)]
        new_parameters.append(param.replace(annotation=new_annotation))

    # Fields that use default_factory show up in inspect.signature with a
    # sentinel default that typer stringifies to "<factory>". When the user
    # does not override such a field via CLI/env, drop it from kwargs so the
    # dataclass constructor invokes the factory normally.
    factory_field_names = {
        f.name for f in dataclasses.fields(dataclass_cls)
        if f.default_factory is not dataclasses.MISSING
    }

    def wrapped(**kwargs):
        for name in factory_field_names:
            if kwargs.get(name) == "<factory>":
                del kwargs[name]
        data = dataclass_cls(**kwargs)
        print(f"Execute command with args: {data}")
        return func(data)

    wrapped.__signature__ = signature.replace(parameters=new_parameters)
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__
    wrapped.__qualname__ = func.__qualname__

    return wrapped


# unit test
if __name__ == "__main__":
    from typer.testing import CliRunner

    @dataclasses.dataclass
    class DemoArgs:
        name: str
        count: int = 1

    app = typer.Typer()

    @app.command()
    @dataclass_cli
    def main(args: DemoArgs):
        print(f"{args.name}|{args.count}")

    runner = CliRunner()

    res1 = runner.invoke(app, [], env={"SLIME_SCRIPT_NAME": "EnvName", "SLIME_SCRIPT_COUNT": "10"})
    print(f"{res1.stdout=}")
    assert res1.exit_code == 0
    assert "EnvName|10" in res1.stdout.strip()

    res2 = runner.invoke(app, ["--count", "999"], env={"SLIME_SCRIPT_NAME": "EnvName"})
    print(f"{res2.stdout=}")
    assert res2.exit_code == 0
    assert "EnvName|999" in res2.stdout.strip()

    print("✅ All Tests Passed!")
