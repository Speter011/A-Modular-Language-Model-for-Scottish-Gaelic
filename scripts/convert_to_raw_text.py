from pathlib import Path
import re

# Folders
project_root = Path(__file__).resolve().parent.parent
input_folder = project_root / "data" / "raw"
print(input_folder)
output_folder = project_root / "data" / "clean"
print(output_folder)

# Match a token ending in an ARCOSG tag, e.g. bha/V-s or e/Pp3sm
TAG_PATTERN = re.compile(r"/[A-Za-z][A-Za-z0-9_-]*$")

# Remove standalone annotation markers, e.g. [1], [3], [?]
ANNOTATION_PATTERN = re.compile(r"^\[[^\]]*\]$")


def clean_arcosg_text(text):
    cleaned_lines = []

    for line in text.splitlines():
        cleaned_tokens = []

        for token in line.split():
            # Remove the POS tag
            token = TAG_PATTERN.sub("", token)

            # Skip empty tokens and annotation markers
            if not token or ANNOTATION_PATTERN.fullmatch(token):
                continue

            cleaned_tokens.append(token)

        if cleaned_tokens:
            cleaned_lines.append(" ".join(cleaned_tokens))

    return "\n".join(cleaned_lines)


# Process every .txt file, including files in subfolders
for input_file in input_folder.rglob("*.txt"):
    relative_path = input_file.relative_to(input_folder)
    output_file = output_folder / relative_path

    # Create the necessary output subfolders
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read, clean, and save
    text = input_file.read_text(encoding="utf-8-sig")
    cleaned_text = clean_arcosg_text(text)

    # save files using utf-8
    output_file.write_text(cleaned_text, encoding="utf-8")

    print(f"Cleaned: {relative_path}")

print("Finished cleaning ARCOSG files.")