import spacy
import json
from spacy.cli.package import package
from pathlib import Path

# paths for trained model
# output for: packaged model, name and version
model_path = Path("../training/output/model-best")
output_path = Path("../training/output/")
package_name = "core_arcosg_sm"
package_version = "0.0.1"
lang_code = "gd"

# Ensure output folder exists
output_path.mkdir(exist_ok=True)

# edit json file to standardize the config
meta_file = model_path / "meta.json"
with open(meta_file, "r+", encoding="utf8") as f:
    meta = json.load(f)
    meta["lang"] = lang_code
    meta["name"] = package_name
    meta["version"] = package_version
    meta["description"] = "Custom Scottish Gaelic pipeline (ARCOSG)"
    f.seek(0)
    json.dump(meta, f, indent=4)
    f.truncate()

# package the model using meta.json
package(
    model_path,
    output_path,
    meta_path=None,
    name=package_name,
    version=package_version,
    create_wheel=True,
    force=True
)


# path to the wheel file is inside the dist folder.
dist_dir = output_path / f"{lang_code}_{package_name}-{package_version}" / "dist"
if dist_dir.exists():
    wheel_files = list(dist_dir.glob("*.whl"))
else:
    wheel_files = list(output_path.rglob("*.whl"))


# success and usage guide or error.
if wheel_files:
    wheel_file = wheel_files[0].resolve()
    print(f"\nPackage created in {output_path}")
    print("Run the following command to install it:\n")
    print(f'    python -m pip install "{wheel_file}"\n')
    print(f"It can then be loaded in python using:\n")
    print(f"    import spacy")
    print(f"    nlp = spacy.load({lang_code}_{package_name})\n")
else:
    print("Could not find wheel file. Check the packaging step again in")