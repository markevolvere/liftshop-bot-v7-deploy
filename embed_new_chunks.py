#!/usr/bin/env python3
"""
embed_new_chunks.py — Cloud Shell script.

Embeds all chunks and facts that have NULL embeddings using Voyage AI.
Run AFTER applying ocr_documents_chunks.sql and ocr_facts.sql.

Usage:
    export DATABASE_URL=$(az webapp config appsettings list \
        --name liftshop-teams-bot --resource-group MjeanesResourceGroup \
        --query "[?name=='DATABASE_URL'].value" -o tsv)
    export VOYAGE_API_KEY=$(az webapp config appsettings list \
        --name liftshop-teams-bot --resource-group MjeanesResourceGroup \
        --query "[?name=='VOYAGE_API_KEY'].value" -o tsv)
    pip install voyageai psycopg2-binary --user -q
    python3 embed_new_chunks.py
"""
import os, sys, time, logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATABASE_URL   = os.environ.get("DATABASE_URL", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL   = "voyage-3"
BATCH_SIZE     = 64

for var, val in [("DATABASE_URL", DATABASE_URL), ("VOYAGE_API_KEY", VOYAGE_API_KEY)]:
    if not val:
        sys.exit(f"ERROR: {var} not set.")

try:
    import voyageai, psycopg2
except ImportError as e:
    sys.exit(f"Missing: {e}\nRun: pip install voyageai psycopg2-binary --user -q")

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
voyage.embed(["warmup"], model=VOYAGE_MODEL, input_type="query")
log.info("Voyage AI ✅")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()
log.info("Database ✅")


def emb_str(emb):
    return "[" + ",".join(f"{v:.8f}" for v in emb) + "]"


def embed_batch(texts):
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:120_000] for t in texts[i:i+BATCH_SIZE]]
        resp = voyage.embed(batch, model=VOYAGE_MODEL, input_type="document")
        results.extend(resp.embeddings)
        if i > 0:
            time.sleep(0.1)
    return results


# ── CHUNKS ────────────────────────────────────────────────────────────────────
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
null_chunks = cur.fetchone()[0]
log.info("Chunks with NULL embedding: %d", null_chunks)

offset = 0
chunks_done = 0
while True:
    cur.execute("""
        SELECT id, content FROM chunks
        WHERE embedding IS NULL
        ORDER BY id
        LIMIT %s OFFSET %s
    """, (BATCH_SIZE, offset))
    rows = cur.fetchall()
    if not rows:
        break

    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    try:
        embeddings = embed_batch(texts)
    except Exception as e:
        log.warning("Batch failed, skipping: %s", e)
        offset += BATCH_SIZE
        continue

    for row_id, emb in zip(ids, embeddings):
        cur.execute("UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                    (emb_str(emb), row_id))
    conn.commit()
    chunks_done += len(rows)
    log.info("Chunks: %d / %d embedded", chunks_done, null_chunks)
    offset += BATCH_SIZE

log.info("✅ Chunks complete — %d embedded", chunks_done)


# ── FACTS ─────────────────────────────────────────────────────────────────────
cur.execute("SELECT COUNT(*) FROM facts WHERE embedding IS NULL")
null_facts = cur.fetchone()[0]
log.info("Facts with NULL embedding: %d", null_facts)

offset = 0
facts_done = 0
while True:
    cur.execute("""
        SELECT id, content FROM facts
        WHERE embedding IS NULL
        ORDER BY id
        LIMIT %s OFFSET %s
    """, (BATCH_SIZE, offset))
    rows = cur.fetchall()
    if not rows:
        break

    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    try:
        embeddings = embed_batch(texts)
    except Exception as e:
        log.warning("Batch failed, skipping: %s", e)
        offset += BATCH_SIZE
        continue

    for row_id, emb in zip(ids, embeddings):
        cur.execute("UPDATE facts SET embedding = %s::vector WHERE id = %s",
                    (emb_str(emb), row_id))
    conn.commit()
    facts_done += len(rows)
    log.info("Facts: %d / %d embedded", facts_done, null_facts)
    offset += BATCH_SIZE

log.info("✅ Facts complete — %d embedded", facts_done)
log.info("=" * 60)
log.info("ALL DONE — %d chunks + %d facts embedded", chunks_done, facts_done)
cur.close()
conn.close()
