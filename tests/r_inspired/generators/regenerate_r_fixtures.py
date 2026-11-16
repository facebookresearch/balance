# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate independently owned r_inspired fixture groups using R only.

This standard-library build helper never imports balance or computes estimates.
Group manifests define ownership and recipe selection. Frozen JSON is read only
for explicit original input/configuration sections, never for expected estimates.
Only selected groups' declared data/notices/manifests reach the final directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

INSPIRED = "fbcode/core_stats/balance/parent_balance/tests/r_inspired"
GENERATORS = INSPIRED + "/generators"
FIXTURES = INSPIRED + "/fixtures"
RECIPE_DEPENDENCIES = {
    "generate_survey_package_fixtures.R": ("survey", "MatchIt", "Matrix"),
    "generate_matrixstats_fixtures.R": ("matrixStats",),
    "generate_subset_quantile_fixtures.R": ("matrixStats", "survey", "Matrix"),
    "generate_glmnet_fixtures.R": (
        "glmnet",
        "Matrix",
        "Rcpp",
        "RcppEigen",
        "foreach",
        "iterators",
        "shape",
    ),
    "generate_matchit_diagnostics.R": ("MatchIt", "survey", "Matrix"),
}
GLM_RECIPE = "generate_glmnet_fixtures.R"
GLM_DATA = "third-party/R/glmnet/data/BinomialExample.rda"
FOLD_PAIR_SHA256 = "09a2569e19ce8a74004f0a0e94eff7fac359654c8f1f7bec0fa66280c8aee4dd"
R_PREFLIGHT = """
a <- commandArgs(TRUE); root <- a[[1]]; needed <- a[-1]
for (p in needed) {
 if (!requireNamespace(p, quietly=TRUE)) stop(paste("Missing R dependency:",p))
 d <- file.path(root,"third-party/R",p,"DESCRIPTION")
 if (p != "Matrix" && file.exists(d) && packageVersion(p) != numeric_version(read.dcf(d)[1,"Version"]))
  stop(paste("Version mismatch:",p))
}
write.csv(data.frame(key=c("R.version","R.share",needed),value=c(R.version.string,
 R.home("share"),vapply(needed,function(p) as.character(packageVersion(p)),character(1)))),
 stdout(),row.names=FALSE)
"""


@dataclass(frozen=True)
class Group:
    name: str
    directory: Path
    manifest: dict
    recipes: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return rows


def read_dcf(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    key = ""
    for line in path.read_text().splitlines():
        if not line:
            continue
        if line[0].isspace() and key:
            fields[key] += " " + line.strip()
        else:
            key, separator, value = line.partition(":")
            if not separator:
                raise ValueError(f"Malformed DESCRIPTION: {path}")
            fields[key] = value.strip()
    return fields


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_notice(source: Path, destination: Path) -> str:
    text = (
        "\n".join(
            line.rstrip(" \t") for line in source.read_text().splitlines()
        ).rstrip("\n")
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return sha256(destination)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recipe_names(manifest: dict) -> tuple[str, ...]:
    values = manifest.get("recipes")
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("Every group manifest needs an explicit recipes list")
    names = [relative_path(v).name for v in values]
    if any(name not in RECIPE_DEPENDENCIES for name in names):
        raise ValueError(f"Unknown R recipe: {values}")
    return tuple(name for name in RECIPE_DEPENDENCIES if name in names)


def discover_groups(fixtures: Path, requested: list[str] | None = None) -> list[Group]:
    # One immediate directory level; unselected groups impose no schema/data requirements.
    available = {
        d.name: d
        for d in sorted(fixtures.iterdir())
        if d.is_dir() and (d / "manifest.json").is_file()
    }
    if not available:
        raise ValueError(f"No fixture-group manifests found under {fixtures}")
    wanted = set(requested) if requested else set(available)
    unknown = wanted - available.keys()
    if unknown:
        raise ValueError(
            f"Unknown groups {sorted(unknown)}; available: {sorted(available)}"
        )
    selected = []
    for name in sorted(wanted):
        directory = available[name]
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key in ("fixture_sha256", "license_sha256", "source_sha256"):
            if not isinstance(manifest.get(key), dict):
                raise ValueError(f"{manifest_path}: missing {key} ownership map")
            for path in manifest[key]:
                relative_path(path)
        scope = manifest.get("glmnet_scope")
        if scope is not None and scope not in ("scaler", "cv", "ipw"):
            raise ValueError(f"Unknown glmnet_scope in {manifest_path}: {scope}")
        recipes = recipe_names(manifest)
        if scope is not None and GLM_RECIPE not in recipes:
            raise ValueError(f"GLM scope without GLM recipe: {directory.name}")
        selected.append(Group(name, directory, manifest, recipes))
    return selected


def selected_recipes(groups: list[Group]) -> tuple[str, ...]:
    wanted = {name for group in groups for name in group.recipes}
    return tuple(name for name in RECIPE_DEPENDENCIES if name in wanted)


def required_packages(recipes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({p for name in recipes for p in RECIPE_DEPENDENCIES[name]}))


def glm_fixture_name(group: Group) -> str:
    names = [
        name for name in group.manifest["fixture_sha256"] if name.endswith(".json")
    ]
    selected = group.manifest.get("glmnet_fixture")
    if selected is not None and selected in names:
        return selected
    if len(names) != 1:
        raise ValueError(f"{group.name}: identify a unique GLM JSON fixture")
    return names[0]


def input_configuration(group: Group) -> dict:
    frozen = json.loads((group.directory / glm_fixture_name(group)).read_text())
    # Do not consume cases, scalers, predictions, weights, or CV expectations.
    configuration = {
        key: frozen[key] for key in ("columns", "x", "y", "foldids_sample_first")
    }
    if configuration["columns"] != [f"x{i:02d}" for i in range(1, 31)]:
        raise ValueError(f"{group.name}: unexpected input columns")
    x, y, folds = (
        configuration["x"],
        configuration["y"],
        configuration["foldids_sample_first"],
    )
    if (
        len(x) != 100
        or any(len(row) != 30 for row in x)
        or len(y) != 100
        or set(y) != {0, 1}
    ):
        raise ValueError(f"{group.name}: invalid original GLM inputs")
    if len(folds) != 100 or any(
        not isinstance(f, int) or isinstance(f, bool) for f in folds
    ):
        raise ValueError(f"{group.name}: missing/invalid captured folds")
    return configuration


def fold_pairs(configuration: dict) -> list[list[int]]:
    y = configuration["y"]
    order = [i + 1 for i, v in enumerate(y) if v == 1] + [
        i + 1 for i, v in enumerate(y) if v == 0
    ]
    pairs = [
        [row, fold] for row, fold in zip(order, configuration["foldids_sample_first"])
    ]
    digest = hashlib.sha256(
        json.dumps(pairs, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != FOLD_PAIR_SHA256:
        raise ValueError(
            "Configuration differs from the original captured fold assignments"
        )
    return pairs


def configurations(groups: list[Group]) -> tuple[dict | None, list[list[int]]]:
    values = [
        input_configuration(g)
        for g in groups
        if g.manifest.get("glmnet_scope") is not None
    ]
    if not values:
        if GLM_RECIPE in selected_recipes(groups):
            raise ValueError(
                "Selected GLM recipe requires an explicit group input/configuration section"
            )
        return None, []
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(
            "Selected GLM groups disagree on original inputs/configuration"
        )
    return first, fold_pairs(first)


def stage_folds(raw: Path, pairs: list[list[int]], supplied: str | None = None) -> None:
    if supplied:
        rows = read_csv(Path(supplied))
        if (
            set(rows[0]) != {"row", "fold"}
            or [[int(r["row"]), int(r["fold"])] for r in rows] != pairs
        ):
            raise ValueError(
                "Explicit foldids.csv disagrees with captured group configuration"
            )
    with (raw / "foldids.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "fold"])
        writer.writerows(pairs)


def source_inventory(repo: Path, groups: list[Group]) -> dict[str, dict[str, str]]:
    inventories = {}
    for group in groups:
        names = set(group.manifest["source_sha256"])
        names.update(f"{GENERATORS}/{name}" for name in group.recipes)
        inventories[group.name] = {
            name: sha256(repo / relative_path(name)) for name in sorted(names)
        }
    return inventories


def assert_sources_unchanged(
    repo: Path, inventories: dict[str, dict[str, str]]
) -> None:
    for files in inventories.values():
        for name, digest in files.items():
            if sha256(repo / relative_path(name)) != digest:
                raise ValueError(f"Source changed during regeneration: {name}")


def r_environment(libraries: list[str] | None) -> dict[str, str]:
    env = dict(os.environ)
    if libraries:
        paths = [p for value in libraries for p in value.split(os.pathsep)]
        if any(not Path(p).is_dir() for p in paths):
            raise FileNotFoundError("Every --r-library directory must exist")
        env["R_LIBS_USER"] = os.pathsep.join(paths)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    return env


def runtime_metadata(
    rscript: str, repo: Path, env: dict, raw: Path, recipes: tuple[str, ...]
) -> dict:
    packages = required_packages(recipes)
    result = subprocess.run(
        [rscript, "--vanilla", "-e", R_PREFLIGHT, str(repo), *packages],
        env=env,
        cwd=raw,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "R dependency/runtime preflight failed:\n" + result.stderr.strip()
        )
    path = raw / "runtime_versions.csv"
    path.write_text(result.stdout)
    return {r["key"]: r["value"] for r in read_csv(path)}


def run_recipes(
    repo: Path, raw: Path, env: dict, rscript: str, recipes: tuple[str, ...]
) -> None:
    for name in recipes:
        source = repo / GENERATORS / name
        root = repo / "third-party/R/glmnet" if name == GLM_RECIPE else repo
        log_path = raw / (name + ".log")
        with log_path.open("w") as log:
            result = subprocess.run(
                [rscript, "--vanilla", str(source), str(root), str(raw)],
                env=env,
                cwd=raw,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            tail = "\n".join(log_path.read_text().splitlines()[-35:])
            raise ValueError(f"{name} failed with exit {result.returncode}:\n{tail}")


def finite_number(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite R output: {name}")
    return value


def glm_inputs(raw: Path, expected: dict) -> tuple[list[dict[str, str]], dict]:
    rows = read_csv(raw / "inputs.csv")
    actual = {
        "columns": [c for c in rows[0] if c.startswith("x")],
        "x": [[finite_number(r, c) for c in expected["columns"]] for r in rows],
        "y": [int(r["y"]) for r in rows],
    }
    if [int(r["row"]) for r in rows] != list(range(1, 101)):
        raise ValueError("Unexpected R upstream row identifiers")
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError(
            "R loaded inputs differ from the group original input sections"
        )
    actual["foldids_sample_first"] = expected["foldids_sample_first"]
    return rows, actual


def glm_source(
    repo: Path, runtime: dict, inventories: dict, group: Group, compatibility: str
) -> dict:
    prefix = repo / "third-party/R/glmnet"
    description = read_dcf(prefix / "DESCRIPTION")
    upstream = None
    for expression in ast.parse((prefix / "METADATA.bzl").read_text()).body:
        if isinstance(expression, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "METADATA" for t in expression.targets
        ):
            upstream = ast.literal_eval(expression.value)["upstream_url"]
            break
    if upstream is None:
        raise ValueError("Missing literal glmnet upstream metadata declaration")
    recipe = f"{GENERATORS}/{GLM_RECIPE}"
    return {
        "package": "glmnet",
        "version": description["Version"],
        "installed_version": runtime["glmnet"],
        "declared_license": description["License"],
        "upstream_archive": upstream,
        "source_notice_files": [
            "third-party/R/glmnet/DESCRIPTION",
            "third-party/R/glmnet/METADATA.bzl",
            "third-party/R/glmnet/R/data.R",
            "third-party/R/glmnet/man/BinomialExample.Rd",
        ],
        "vendored_path": GLM_DATA,
        "sha256": sha256(repo / GLM_DATA),
        "dataset_description": "Upstream synthetic binomial example; exact stored values, not regenerated.",
        "semantic_mapping": "y=1 interpreted as sample membership, y=0 as target; unit design weights added because upstream dataset has no weights.",
        "r_recipe": "../../generators/" + GLM_RECIPE,
        "r_recipe_sha256": inventories[group.name][recipe],
        "r_thresh": 1e-14,
        "runtime": runtime["R.version"],
        "historical_runtime_build": False,
        "build_compatibility": compatibility,
        "author_from_DESCRIPTION": description.get(
            "Author", description.get("Authors@R", "")
        ),
        "dependencies": {
            p: runtime[p] for p in required_packages((GLM_RECIPE,)) if p != "glmnet"
        },
        "fixture_scope": group.manifest["glmnet_scope"],
    }


def append_glm_scope(
    raw: Path, fixture: dict, scope: str, rows: list[dict[str, str]]
) -> None:
    if scope == "scaler":
        fixture["scalers"] = {}
        for balanced in ("FALSE", "TRUE"):
            scales = read_csv(raw / f"scaler_balanced_{balanced}.csv")
            if [r["column"] for r in scales] != fixture["columns"]:
                raise ValueError("R scaler column mismatch")
            fixture["scalers"][balanced] = {
                k: [finite_number(r, k) for r in scales] for k in ("mean", "scale")
            }
        return
    fixture["cases"] = {}
    lambdas = (1,) if scope == "cv" else (1, 0.1)
    for balanced in ("FALSE", "TRUE"):
        for lam in lambdas:
            key = f"lambda_{lam:g}_balanced_{balanced}"
            case = {"lambda": lam, "balance_classes": balanced == "TRUE"}
            if scope == "cv":
                folds = read_csv(raw / (key + "_cv_folds.csv"))
                metrics = read_csv(raw / (key + "_cv_metrics.csv"))
                if [int(r["fold"]) for r in folds] != list(range(10)) or len(
                    metrics
                ) != 1:
                    raise ValueError(f"Incomplete R CV outputs: {key}")
                case.update(
                    cv_mean=finite_number(metrics[0], "mean"),
                    cv_se=finite_number(metrics[0], "se"),
                    cv_fold_deviance=[finite_number(r, "deviance") for r in folds],
                )
            else:
                values = read_csv(raw / (key + ".csv"))
                if [(r["row"], r["y"]) for r in values] != [
                    (r["row"], r["y"]) for r in rows
                ]:
                    raise ValueError(f"R prediction row/label mismatch: {key}")
                case.update(
                    probability=[finite_number(r, "probability") for r in values],
                    sample_weight=[
                        finite_number(r, "weight") for r in values if r["y"] == "1"
                    ],
                )
            fixture["cases"][key] = case


def assemble_glmnet(
    repo: Path,
    raw: Path,
    destination: Path,
    runtime: dict,
    inventories: dict,
    group: Group,
    configuration: dict,
    compatibility: str,
) -> None:
    rows, inputs = glm_inputs(raw, configuration)
    fixture = {
        "source": glm_source(repo, runtime, inventories, group, compatibility),
        "columns": inputs["columns"],
        "x": inputs["x"],
        "y": inputs["y"],
    }
    append_glm_scope(raw, fixture, group.manifest["glmnet_scope"], rows)
    fixture["foldids_sample_first"] = inputs["foldids_sample_first"]
    write_json(destination, fixture)


def sanitize_matchit(path: Path) -> None:
    rows = read_csv(path)
    excluded = {
        "expected_balance_asmd_target",
        "expected_balance_ks",
        "matchit_ecdf_mean",
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[k for k in rows[0] if k not in excluded],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def notice_source(
    repo: Path, group: Group, name: str, runtime: dict, mode: str
) -> Path:
    path = relative_path(name)
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ("licenses", "packages"):
        return repo / "third-party/R" / Path(*parts[2:])
    if (
        len(parts) == 2
        and parts[0] == "licenses"
        and path.suffix == ".txt"
        and mode == "executed R"
    ):
        return Path(runtime["R.share"]) / "licenses" / path.stem
    # Assembly-only tests preserve the already captured canonical source notices.
    return group.directory / path


def assemble_group(
    repo: Path,
    raw: Path,
    output: Path,
    runtime: dict,
    inventories: dict,
    group: Group,
    configuration: dict | None,
    source_revision: str,
    compatibility: str | None,
    mode: str,
) -> dict:
    destination = output / group.name
    destination.mkdir()
    note = (
        compatibility
        if compatibility is not None
        else group.manifest.get(
            "compatibility",
            "Installed R build provenance not supplied; no claim of pristine historical-runtime execution.",
        )
    )
    glm_name = glm_fixture_name(group) if group.manifest.get("glmnet_scope") else None
    for name in group.manifest["fixture_sha256"]:
        target = destination / relative_path(name)
        if name == glm_name:
            assemble_glmnet(
                repo, raw, target, runtime, inventories, group, configuration, note
            )
        else:
            copy_file(raw / relative_path(name), target)
            if Path(name).name in (
                "lalonde_matchit_expected.csv",
                "lalonde_subclass_within_ks_expected.csv",
            ):
                sanitize_matchit(target)
    license_hashes = {
        name: copy_notice(
            notice_source(repo, group, name, runtime, mode),
            destination / relative_path(name),
        )
        for name in group.manifest["license_sha256"]
    }
    manifest = dict(group.manifest)
    needed = required_packages(group.recipes)
    manifest.update(
        source_revision=source_revision,
        source_sha256=inventories[group.name],
        fixture_sha256={
            name: sha256(destination / relative_path(name))
            for name in group.manifest["fixture_sha256"]
        },
        license_sha256=license_hashes,
        r_runtime=runtime.get("R.version", group.manifest.get("r_runtime")),
        package_versions={
            k: v
            for k, v in runtime.items()
            if k in needed or k in ("R.version", "R.share")
        },
        recipes=list(group.recipes),
        compatibility=note,
        r_execution_mode=mode,
        assembler_sha256=sha256(Path(__file__).resolve()),
    )
    if group.manifest.get("glmnet_scope"):
        manifest["captured_fold_pair_sha256"] = FOLD_PAIR_SHA256
    write_json(destination / "manifest.json", manifest)
    return {
        "fixtures": len(manifest["fixture_sha256"]),
        "notices": len(license_hashes),
        "scope": group.manifest.get("glmnet_scope"),
    }


def assemble_selected(
    repo: Path,
    raw: Path,
    output: Path,
    groups: list[Group],
    runtime: dict,
    inventories: dict,
    source_revision: str,
    compatibility: str | None = None,
    mode: str = "captured R outputs (assembly only)",
) -> dict:
    configuration, pairs = configurations(groups)
    if pairs:
        recorded = [
            [int(r["row"]), int(r["fold"])] for r in read_csv(raw / "foldids.csv")
        ]
        if recorded != pairs:
            raise ValueError("Raw foldids.csv differs from group configuration")
    return {
        group.name: assemble_group(
            repo,
            raw,
            output,
            runtime,
            inventories,
            group,
            configuration,
            source_revision,
            compatibility,
            mode,
        )
        for group in groups
    }


def revision(repo: Path, supplied: str | None) -> str:
    if supplied:
        return supplied
    value = subprocess.run(
        ["sl", "log", "-r", ".", "-T", "{node}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Could not determine source revision; pass --source-revision")
    return value


def regenerate(args: argparse.Namespace) -> dict:
    repo, out = Path(args.fbsource).resolve(), Path(args.outdir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(
            "Destination must be absent or empty; refusing stale fixture mixing"
        )
    groups = discover_groups(repo / FIXTURES, args.group)
    recipes = selected_recipes(groups)
    configuration, pairs = configurations(groups)
    inventories = source_inventory(repo, groups)
    source_revision = revision(repo, args.source_revision)
    executable = shutil.which(args.rscript) if recipes else None
    if recipes and executable is None:
        raise FileNotFoundError(f"Rscript executable not found: {args.rscript}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".r-inspired-", dir=out.parent
    ) as temporary:
        work = Path(temporary)
        raw = work / "raw"
        bank = work / "groups"
        raw.mkdir()
        bank.mkdir()
        if pairs:
            stage_folds(raw, pairs, args.foldids)
        if recipes:
            env = r_environment(args.r_library)
            runtime = runtime_metadata(executable, repo, env, raw, recipes)
            run_recipes(repo, raw, env, executable, recipes)
        else:
            runtime = {}
        summary = assemble_selected(
            repo,
            raw,
            bank,
            groups,
            runtime,
            inventories,
            source_revision,
            args.compatibility_note,
            mode="executed R" if recipes else "no R recipes required",
        )
        assert_sources_unchanged(repo, inventories)
        if out.exists():
            out.rmdir()
        os.replace(bank, out)
    return {"outdir": str(out), "groups": summary, "recipes_executed": list(recipes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbsource", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--group",
        action="append",
        help="Optional/repeatable manifest group; omitted selects all available groups",
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--r-library",
        action="append",
        help="Library path, path-separated list, or repeated flag",
    )
    parser.add_argument(
        "--foldids",
        help="Optional captured-fold override; must match group input/configuration",
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--compatibility-note")
    try:
        result = regenerate(parser.parse_args())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Regeneration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
