"""Render params/parameters.toml as a markdown table for Appendix A, keeping the
comment that says where each number comes from. Run after any change to the
parameter file so the appendix cannot drift from the code.

Method: read the parameter file line by line (not with a TOML parser, which
would drop the comments). Each "key = value  # note" line becomes a table row
with its section. An indented comment line right under an entry continues that
entry's note. Then count how many notes start with measured, chosen or derived.

Run:  pixi run python notebooks/12_parameter_table.py
Writes APPENDIX_A_parameters.md
"""
import re
from pathlib import Path

# ---- 1. read the parameter file as plain lines ----
HERE = Path(__file__).resolve().parent.parent
src = (HERE / "params" / "parameters.toml").read_text(encoding="utf-8").splitlines()

# ---- 2. turn the lines into rows: [section, key, value, note] ----
# section is the [header] currently in force; last is the row an indented comment would continue.
rows, section, last = [], "", None     # last: the row an indented comment line continues
for line in src:
    s = line.rstrip()
    if not s.strip():
        continue
    m = re.match(r"^\[([^\]]+)\]\s*(#.*)?$", s.strip())
    if m:                               # a section header (its own comment, if any, is not a row's)
        section, last = m.group(1), None
        continue
    if s.lstrip().startswith("#"):
        # an indented comment line continues the note of the entry just above it;
        # a comment at the start of a line is about the section, not a value
        if last is not None and s[0].isspace():
            last[3] += " " + s.strip().lstrip("#").strip()
        else:
            last = None
        continue
    # a "key = value  # note" line; the note is optional
    m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.+?)(\s+#\s*(.*))?$", s)
    if m:
        key, value, note = m.group(1), m.group(2).strip(), (m.group(4) or "").strip()
        last = [section, key, value, note]
        rows.append(last)

# ---- 3. write the appendix: a short header, the table, and the count by label ----
kind = {"measured": 0, "chosen": 0, "derived": 0}
out = ["# Appendix A, parameters: every number, with its origin", "",
       "Rendered from `params/parameters.toml` by `notebooks/12_parameter_table.py`. The note on each",
       "line is the one in the file: *measured* means it is in a cited source or fitted from one by a",
       "script in this repository, *chosen* means picked by me with the rule stated, *derived* means",
       "arithmetic on the others. Sources are abbreviated as in the file header (IDAHO = Lyon 2016;",
       "YIN18 = Yin et al. 2018; ECJTU = US 11,214,850 B2; CARESTER = Rollat et al. 2024; AE = American",
       "Elements product page).", "",
       "| section | parameter | value | origin |", "| --- | --- | --- | --- |"]
for section, key, value, note in rows:
    for k in kind:
        if note.lower().startswith(k):
            kind[k] += 1
    # a "|" inside a note would break the markdown table, so it becomes "/"
    out.append(f"| {section} | `{key}` | `{value}` | {note.replace('|', '/')} |")
out += ["", f"Counted by label where the note starts with one: {kind['measured']} measured, {kind['chosen']} chosen, "
        f"{kind['derived']} derived (entries whose note starts otherwise are not counted)."]
(HERE / "APPENDIX_A_parameters.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print(f"wrote APPENDIX_A_parameters.md with {len(rows)} parameters")
