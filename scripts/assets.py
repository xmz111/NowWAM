"""Download and prepare simulator assets."""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, destination, expected):
    """Fetch a public archive atomically, refusing unexpected existing content."""
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f"Checksum mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as source:
                shutil.copyfileobj(source, stream)
            stream.close()
            if sha256(temporary) != expected:
                raise ValueError(f"Download checksum mismatch: {url}")
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def zip_subtree(archive, name):
    """Find a single named root, ignoring the producer's parent directory layout."""
    with zipfile.ZipFile(archive) as source:
        roots = {
            item.filename
            for item in source.infolist()
            if item.is_dir()
            and (item.filename == name + "/" or item.filename.endswith("/" + name + "/"))
        }
    if len(roots) != 1:
        raise ValueError(f"Expected one {name} root in archive, found {len(roots)}")
    return roots.pop()


def extract_zip(archive, destination, prefix, *, missing_only=False):
    """Extract a pinned subtree without traversal, symlinks or silent replacement."""
    destination = destination.resolve()
    selected = []
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink: {item.filename}")
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            target = destination / item.filename[len(prefix) :]
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Archive path escapes destination: {target}")
            selected.append((item, target))
        if not selected or len({p for _, p in selected}) != len(selected):
            raise ValueError("Empty or duplicate archive subtree")
        for item, target in selected:
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"Existing non-file asset: {target}")
                if not missing_only:
                    with source.open(item) as stream:
                        expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    if sha256(target) != expected:
                        raise ValueError(f"Refusing to replace changed asset: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with source.open(item) as member:
                        shutil.copyfileobj(member, stream)
                    stream.close()
                    os.link(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)


def apply_source_patch(root, filename):
    patch = Path(__file__).with_name(filename).resolve()
    command = ["git", "-C", str(root), "apply"]
    if subprocess.run([*command, "--check", str(patch)], capture_output=True).returncode == 0:
        subprocess.run([*command, str(patch)], check=True)
    elif subprocess.run(
        [*command, "--reverse", "--check", str(patch)], capture_output=True
    ).returncode:
        raise ValueError(f"Source differs from expected states for {filename}")


def patch_dial(root):
    expected = "b611fed034c62441413fea2b136ff7c841f382fe"
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"Expected DIAL revision {expected}, got {actual}")
    for name, expected_count in (("multistep_wrapper.py", 3), ("video_recording_wrapper.py", 6)):
        path = root / "gr00t/eval/wrappers" / name
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:gr00t/eval/wrappers/{name}"], text=True
        )
        if original.count("pdb.set_trace()") != expected_count:
            raise ValueError(f"Unexpected debug-hook count in {name}")
        patched = original.replace("pdb.set_trace()", "pass  # Non-interactive evaluation.")
        ast.parse(patched)
        if path.read_text() not in (original, patched):
            raise ValueError(f"Refusing to overwrite unrelated edits: {path}")
        path.write_text(patched)
        print(f"Patched {name}: removed {expected_count} debugger calls")


def prepare_libero(root, config):
    expected = "4976dc30028e805ff8094b55501d532c48fec182"
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"Expected LIBERO-Plus revision {expected}, got {actual}")
    package = root / "libero/libero"
    path = package / "benchmark/__init__.py"
    before = path.read_text()
    old = "init_states = torch.load(init_states_path)"
    new = "init_states = torch.load(init_states_path, weights_only=False)"
    if before.count(old) == 2 and new not in before:
        path.write_text(before.replace(old, new))
    elif before.count(new) != 2 or old in before:
        raise ValueError("Unrecognized LIBERO initial-state loader")
    # The pinned assets contain trusted NumPy objects; never apply this loader
    # to arbitrary untrusted checkpoint files.
    (root / "libero/__init__.py").touch(exist_ok=True)
    paths = {
        "benchmark_root": package,
        "bddl_files": package / "bddl_files",
        "init_states": package / "init_files",
        "datasets": root / "datasets",
        "assets": package / "assets",
    }
    for name in ("bddl_files", "init_states", "assets"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    # JSON is valid YAML and avoids requiring PyYAML in the preparation process.
    content = json.dumps({key: str(path.resolve()) for key, path in paths.items()}, indent=2) + "\n"
    config.mkdir(parents=True, exist_ok=True)
    destination = config / "config.yaml"
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Refusing to replace an existing, different LIBERO config")
    destination.write_text(content)
    print(f"Prepared isolated LIBERO config: {config}")


def patch_robocasa(root):
    expected = "4840e671596f93ca03651524b9f72ffb1aadfeff"
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"Expected RoboCasa revision {expected}, got {actual}")
    patch = Path(__file__).with_name("robocasa-fixtures.patch").resolve()
    command = ["git", "-C", str(root), "apply"]
    if subprocess.run([*command, "--check", str(patch)], capture_output=True).returncode == 0:
        subprocess.run([*command, str(patch)], check=True)
        print("Applied the two reference fixture XML changes")
    elif (
        subprocess.run(
            [*command, "--reverse", "--check", str(patch)], capture_output=True
        ).returncode
        == 0
    ):
        print("Reference fixture changes already present")
    else:
        raise ValueError(
            "Fixture files differ from both expected states; inspect rather than overwrite"
        )
