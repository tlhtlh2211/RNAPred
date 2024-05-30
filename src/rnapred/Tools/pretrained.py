import subprocess

# For a single RNA sequence prediction
rna_sequence = "AACCGGGUCAGGUCCGGAAGGAAGCAGCCCUAA"
subprocess.run(["sincFold", "pred", rna_sequence])

