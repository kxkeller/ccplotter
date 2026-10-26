"""ccplotter/routes/__init__.py -- the ONE shared Flask app instance.

Every routes/*.py module registers its @app.route(...) handlers against this
SAME Flask() object. Blueprints are not used (single-deployment desktop app).

Why the Flask() constructor lives HERE and not in app.py
----------------------------------------------------------
The dev server / replay harness starts app.py via `python app.py`
(scripts/replay_harness.py), which makes Python execute this file as
sys.modules["__main__"] -- NOT as sys.modules["app"]. desktop/ccplotter_app.py
and other scripts instead do `import app as ccapp`, which registers it as
sys.modules["app"]. These are two DIFFERENT dict entries in sys.modules for
the same file, even though only one of them ever actually runs in a given
process.

If `app = Flask(__name__)` stayed in app.py and each routes/*.py module did
`from app import app` to get it, that import statement -- executed from
inside ccplotter.routes.core, itself imported by app.py -- would ask Python
to import a module named "app". When app.py is running as "__main__" (the
`python app.py` case), sys.modules has NO entry named "app" yet, so Python
would import app.py A SECOND TIME under the name "app", re-running every
top-level statement in the file (all the ccplotter.* imports, the printer
singleton connect() call, etc.) and creating a SECOND, DIFFERENT Flask
instance. Every routes/*.py would then register its handlers on that ghost
"app" module's Flask instance, while the ORIGINAL "__main__" instance (the
one whose app.run(...) actually gets called at the bottom of app.py) would
have an EMPTY url_map -- every request would 404. This is a real, well-known
Python footgun (the __main__-vs-imported-module duplicate-import hazard),
not a hypothetical -- it is why this Flask instance is defined here,
in a module ("ccplotter.routes") that has exactly ONE possible sys.modules
key regardless of how the entry point was invoked.

root_path / import_name pinning
--------------------------------
Flask resolves default `template_folder="templates"` / `static_folder=
"static"` (and Flask.send_from_directory's relative-directory argument, used
by serve_work/serve_model_image) relative to `root_path`. Left on its
Flask(__name__) default here, root_path would resolve to this file's own
directory (ccplotter/routes/), NOT the project root where the templates
and vpype.toml actually live, so index.html would stop resolving. Pinned explicitly to the
project root (two parents up from this __init__.py: ccplotter/routes/ ->
ccplotter/ -> project root), the same directory `Flask(__name__)` in app.py
resolves to (its __name__ is either "__main__" or "app", and both resolve
root_path via app.py's own directory).

The templates live in assets/templates/. Flask's DEFAULT
template_folder="templates" (relative to root_path) would 404 index.html in
the dev / replay-harness path (the un-overridden case -- see below), so
template_folder is pinned explicitly to assets/templates alongside root_path.
static_folder is left on Flask's default ("static", which does not exist at
the project root either way -- nothing in this app serves Flask static
files; all static assets are inlined in templates/index.html or served via
the explicit /work and /model_images routes).

import_name is pinned to the literal string "app" (not this module's own
__name__, "ccplotter.routes") so Flask's `app.name` is "app" however the entry
point is run (Flask derives the app name from import_name, special-casing
"__main__" to use the entry script's basename, "app" from app.py). Nothing in
this codebase reads app.name directly.

desktop/ccplotter_app.py explicitly OVERRIDES root_path, template_folder,
and jinja_loader at runtime after import (see that file's own comments on
the frozen-build resource-path problem) -- this module's root_path only
matters for the un-overridden case: `python app.py` / the replay harness.
"""
from pathlib import Path

from flask import Flask

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = Flask("app", root_path=str(_PROJECT_ROOT),
            template_folder=str(_PROJECT_ROOT / "assets" / "templates"))
