from pathlib import Path
root = Path.cwd()
p = root / 'code/make_v2_tables.py'
s = p.read_text()
start = s.index('        name = {"single": FAM_NAMES.get(k[1], k[1])')
end = s.index('        rows.append([name,', start)
s = s[:start] + '''        if k[0] == "theorem":
            name = ("stack, unweighted component square" if k[1] == "inf"
                    else "stack, $J'_\\\\tau$ at $\\\\tau = " + k[1] + "$")
        else:
            name = {"single": FAM_NAMES.get(k[1], k[1]),
                    "paper_norm": "stack, mean relative norm",
                    "paper": "stack, row-relative squared error"}[k[0]]
''' + s[end:]
p.write_text(s)
p = root / '.github/workflows/manuscript-checks.yml'
s = p.read_text().replace('latexmk texlive-latex-recommended texlive-fonts-recommended',
                         'latexmk texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra')
p.write_text(s)
compile((root / 'code/make_v2_tables.py').read_text(), 'make_v2_tables.py', 'exec')
print('Finalized generator labels/Python 3.11 compatibility and TeX dependencies.')
