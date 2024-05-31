from varnaapi import Structure

v = Structure(structure="((((((.((((((........)))))).((((((.......))))))..))))))")
v.add_aux_BP(14, 20, edge5="s", color="#FF00FF")
v.savefig("example.png")