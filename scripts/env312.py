import os
import sys


def activate_local_deps():
    """
    This repo uses a local dependency folder to avoid global/venv dependency drift.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    deps = os.path.join(repo_root, ".pydeps312")
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
    # add src/ so `import rdi...` works
    src_dir = os.path.join(repo_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return deps


if __name__ == "__main__":
    deps = activate_local_deps()
    print(deps)


