import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from numpy.linalg import norm

BASE_PATH = r"C:\Users\Divya Rangumudri\Downloads\dataset_collection"

GENES = {
    "TP53": "tp53_reference_cds.fasta.txt",
    "KRAS": "kras_cds.fasta.txt",
    "EGFR": "egfr_cds.fasta.txt",
    "BRAF": "braf_cds.fasta.txt",
    "ALK": "alk_cds.fasta.txt",
    "PIK3CA": "pik3ca_cds.fasta.txt",
    "STK11": "stk11_cds.fasta.txt",
    "KEAP1": "keap1_cds.fasta.txt",
    "NF1": "nf1_cds.fasta.txt",
    "SMARCA4": "smarca4_cds.fasta.txt",
    "RB1": "rb1_cds.fasta.txt",
    "ATM": "atm_cds.fasta.txt"
}

MAF_FILE = os.path.join(BASE_PATH, "data_mutations.txt")
maf = pd.read_csv(MAF_FILE, sep="\t", comment="#", low_memory=False)


# ---------- Reference ----------
def load_reference_sequence(path):
    return str(SeqIO.read(path, "fasta").seq)


# ---------- Build patient mutated sequence ----------
def create_patient_sequences(reference_seq, gene_df):

    patient_sequences = []

    grouped = gene_df.groupby("Tumor_Sample_Barcode")

    for patient_id, rows in grouped:

        seq_list = list(reference_seq)
        mutation_count = 0

        for _, row in rows.iterrows():

            hgvsc = str(row["HGVSc"])

            if "c." not in hgvsc or ">" not in hgvsc:
                continue

            try:
                mut = hgvsc.split("c.")[1]

                num = ""
                for c in mut:
                    if c.isdigit():
                        num += c
                    else:
                        break

                pos = int(num) - 1
                alt = mut.split(">")[-1][0]

                if pos >= len(seq_list):
                    continue

                if seq_list[pos] != alt:
                    seq_list[pos] = alt
                    mutation_count += 1

            except:
                continue

        if mutation_count > 0:
            patient_sequences.append(
                SeqRecord(Seq("".join(seq_list)), id=str(patient_id))
            )

    return patient_sequences


# ---------- Remove duplicates ----------
def remove_redundant(records):
    unique = {}
    for rec in records:
        seq = str(rec.seq)
        if seq not in unique:
            unique[seq] = rec
    return list(unique.values())


# ---------- Alignment score ----------
def alignment_scores(reference, sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    scores = []
    for rec in sequences:
        scores.append(aligner.score(reference, str(rec.seq)))
    return np.mean(scores) if scores else 0


# ---------- Cosine similarity ----------
def encode(seq):
    map = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    vec=[]
    for b in seq:
        vec.extend(map.get(b,[0,0,0,0]))
    return np.array(vec)

def cosine_similarity_avg(reference, sequences):
    r = encode(reference)
    sims=[]
    for rec in sequences:
        m = encode(str(rec.seq))
        L=min(len(r),len(m))
        sims.append(np.dot(r[:L],m[:L])/(norm(r[:L])*norm(m[:L])))
    return np.mean(sims) if sims else 0


# ---------- MAIN ----------
summary=[]

for gene,fasta in GENES.items():

    print("\nProcessing",gene)

    gene_folder=os.path.join(BASE_PATH,gene)
    ref_path=os.path.join(gene_folder,fasta)

    if not os.path.exists(ref_path):
        continue

    reference=load_reference_sequence(ref_path)

    gene_df=maf[(maf["Hugo_Symbol"]==gene)&(maf["HGVSc"].notna())]

    total_sequences=len(gene_df["Tumor_Sample_Barcode"].unique())

    mutated=create_patient_sequences(reference,gene_df)
    mutated_count=len(mutated)

    unique=remove_redundant(mutated)
    non_redundant=len(unique)
    redundant=mutated_count-non_redundant

    align_score=alignment_scores(reference,unique)
    cos_score=cosine_similarity_avg(reference,unique)

    summary.append([gene,total_sequences,mutated_count,redundant,non_redundant,align_score,cos_score])

    print("Total patients:",total_sequences)
    print("Mutated sequences:",mutated_count)
    print("Redundant:",redundant)
    print("Non-redundant:",non_redundant)
    print("Avg Alignment Score:",round(align_score,2))
    print("Avg Cosine Similarity:",round(cos_score,5))


df=pd.DataFrame(summary,columns=["Gene","Total","Mutated","Redundant","NonRedundant","AlignmentScore","CosineSimilarity"])
df.to_csv(os.path.join(BASE_PATH,"FINAL_SUMMARY.csv"),index=False)

print("\nFINAL_SUMMARY.csv generated")
