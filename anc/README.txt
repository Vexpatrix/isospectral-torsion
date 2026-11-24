Reproducing the torsion certificate
===================================

Keep the verifier and witness together and run this command from their
directory.

    python -B verify_torsion_twins.py 64

The verifier uses only the Python standard library. It was tested with
Python 3.14.2. It checks the polygon geometry, transplantation matrices,
and torsional rigidity bounds. A successful run ends with this line.

    CERTIFIED: tau(A) < tau(B); gap >= 0.000135198

SHA-256 checksums
----------------
019b13e311262dcff72cb137ecdb48cec517d741b732890ecfa423636a332d97  verify_torsion_twins.py
fccbd6e3657b0cff81ea74fc0a4f7eea27e886d9bca40134c0f94e83692f8bf0  torsion-witness-k64.json.gz
