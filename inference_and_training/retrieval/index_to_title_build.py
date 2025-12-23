# build_id_map.py
from pyserini.index.lucene import IndexReader
import json, sys

index_dir   = '/content/drive/MyDrive/mRAG_and_MSRS_source/index'             # path to your Lucene index
out_path    = '/content/drive/MyDrive/mRAG_and_MSRS_source/inference_and_training/evaluation'               # where to save the map (json)
id_field    = "id"                     # stored field name in your docs (adjust if needed)

ir = IndexReader(index_dir)
stats = ir.stats()
n_docs = stats['documents']

id_map = {}  # internal_id -> external_id

for internal_id in range(n_docs):
    doc = ir.doc(internal_id)
    if not doc:
        continue

    # prefer stored field
    ext = doc.get(id_field)
    if not ext:
        # else try raw JSON (requires --storeRaw when indexing)
        raw = doc.raw()
        if raw:
            try:
                ext = json.loads(raw).get("id")
            except Exception:
                pass

    if ext:
        # normalize keys to the exact string form that appears in your verified docs
        key = str(internal_id)                   # typical internal-key choice
        id_map[key] = ext                        # map internal -> external

# optional sanity: assert 1-1
# assert len(id_map) == len(set(id_map.values()))

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(id_map, f, ensure_ascii=False)
print(f"wrote {len(id_map)} ids to {out_path}")
