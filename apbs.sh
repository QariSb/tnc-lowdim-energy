#!/bin/bash
set -euo pipefail

# ============================================================
# USER SETTINGS
# ============================================================

WT_PDB="WT_amber.pdb"

declare -A MUTATIONS=(

  [Y5H]="5-HIS"
  [A8V]="8-VAL"
  [F20Q]="20-GLN"
  [A23Q]="23-GLN"
  [L29Q]="29-GLN"
  [A31S]="31-SER"
  [S37G]="37-GLY"
  [E40A]="40-ALA"
  [V44Q]="44-GLN"
  [M45Q]="45-GLN"
  [L48Q]="48-GLN"
  [Q50R]="50-ARG"
  [L57Q]="57-GLN"
  [E59D]="59-ASP"
  [I61Q]="61-GLN"
  [D67A]="67-ALA"
  [D73A]="73-ALA"
  [D73N]="73-ASN"
  [D75Y]="75-TYR"
  [V79Q]="79-GLN"
  [M81Q]="81-GLN"
  [C84Y]="84-TYR"

)

# ============================================================
# Add calcium back into PQR
# ============================================================

add_calcium_to_pqr () {

PDB="$1"
PQR="$2"

echo ">>> Adding calcium safely into ${PQR}"

# ------------------------------------------------------------
# 1. Extract calcium coordinates from original PDB
# ------------------------------------------------------------

CA_LINE=$(grep "^HETATM" "$PDB" | grep " CA   CA " || true)

if [ -z "$CA_LINE" ]; then
    echo "WARNING: No calcium found in $PDB"
    return
fi

X=$(echo "$CA_LINE" | cut -c31-38 | xargs)
Y=$(echo "$CA_LINE" | cut -c39-46 | xargs)
Z=$(echo "$CA_LINE" | cut -c47-54 | xargs)

# ------------------------------------------------------------
# 2. Remove existing END lines safely
# ------------------------------------------------------------

TMPFILE="${PQR}.tmp"

grep -v '^END' "$PQR" > "$TMPFILE"

# ------------------------------------------------------------
# 3. Append calcium with strict PQR formatting
# ------------------------------------------------------------

printf "HETATM%5d %-4s %-3s %1s%4d    %8.3f%8.3f%8.3f %7.4f %7.4f\n" \
9999 CA CAL A 999 $X $Y $Z 2.0000 1.7000 >> "$TMPFILE"

# ------------------------------------------------------------
# 4. Add END as final line
# ------------------------------------------------------------

echo "END" >> "$TMPFILE"

# ------------------------------------------------------------
# 5. Replace original file
# ------------------------------------------------------------

mv "$TMPFILE" "$PQR"

}

# ============================================================
# Create APBS input
# ============================================================

create_apbs_input () {

SYSTEM=$1
PQR=${SYSTEM}.pqr
INFILE=${SYSTEM}.in

cat <<EOF > ${INFILE}
read
    mol pqr ${PQR}
end

elec
    mg-auto

    dime 161 161 161

    cglen 80.0 80.0 80.0
    fglen 40.0 40.0 40.0

    cgcent mol 1
    fgcent mol 1

    mol 1

    lpbe
    bcfl sdh

    pdie 2.0
    sdie 78.5

    chgm spl2
    srfm smol

    srad 1.4
    swin 0.3
    sdens 10.0

    temp 298.15

    calcenergy total
    calcforce no

    write pot dx ${SYSTEM}_potential

end

quit
EOF

}

# ============================================================
# Run APBS pipeline
# ============================================================

run_apbs () {

SYSTEM=$1

echo ">>> Electrostatics for ${SYSTEM}"

cd "${SYSTEM}"

pdb2pqr --ff=AMBER --with-ph=7.0 --keep-chain \
"${SYSTEM}_amber.pdb" "${SYSTEM}.pqr"

# restore calcium ion
add_calcium_to_pqr "${SYSTEM}_amber.pdb" "${SYSTEM}.pqr"

create_apbs_input "${SYSTEM}"

apbs "${SYSTEM}.in"

cd ..
}

# ============================================================
# WT
# ============================================================

echo ">>> Preparing WT"

mkdir -p WT
cp "${WT_PDB}" WT/WT_amber.pdb

run_apbs WT

# ============================================================
# MUTANTS
# ============================================================

for NAME in "${!MUTATIONS[@]}"; do

echo ">>> Preparing mutant ${NAME}"

mkdir -p "${NAME}"

pdb4amber \
-i "${WT_PDB}" \
-o "${NAME}/${NAME}_amber.pdb" \
-m "${MUTATIONS[$NAME]}" \
--reduce \
--add-missing-atoms

run_apbs "${NAME}"

done

echo "ALL DONE — ML-ready electrostatic maps generated."
