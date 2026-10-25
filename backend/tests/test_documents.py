"""S0: extraction, chunking, FAISS retrieval, and the retrieved passages flowing through S1–S5."""
import csv
import io

import pytest

from optimus import documents
from optimus.pipeline import analyze_sync
from optimus.stages import s0_retrieve


def _pdf(pages: list[list[str]]) -> bytes:
    """A minimal text PDF (Helvetica, one line per string)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        stream = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    body, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(body)
    body += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    body += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return body + f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


LEAVE_PDF = _pdf([
    ["Acme Corp Employee Leave Policy 2026.", "Full-time employees receive 24 days of paid annual leave per year.",
     "Leave requests must be submitted at least 14 days in advance through the HR portal."],
    ["Sick leave: employees get 12 days of paid sick leave.",
     "A medical certificate is required for sick leave longer than 3 consecutive days."],
])


def _docx() -> bytes:
    import docx

    d = docx.Document()
    d.add_heading("Project Falcon status report", 0)
    d.add_heading("Timeline", 1)
    d.add_paragraph("The beta launch moved from August to 14 October 2026 because the payment gateway "
                    "certification took longer than planned.")
    d.add_heading("Budget", 1)
    d.add_paragraph("We have spent 1.8 million rupees of the 2.5 million rupee budget.")
    t = d.add_table(rows=2, cols=2)
    for r, (a, b) in enumerate([("Item", "Cost"), ("Cloud hosting", "900000")]):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _pptx() -> bytes:
    from pptx import Presentation

    p = Presentation()
    for title, body in [("Q3 Sales Review", "Revenue grew 18% quarter over quarter\nThe north region led with 4.1 crore"),
                        ("Risks", "Two key accounts are up for renewal in November")]:
        s = p.slides.add_slide(p.slide_layouts[1])
        s.shapes.title.text, s.placeholders[1].text = title, body
    buf = io.BytesIO()
    p.save(buf)
    return buf.getvalue()


def _csv() -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerows([["order_id", "customer", "city", "status"], [1001, "Priya", "Pune", "delivered"],
                 [1002, "Kabir", "Pune", "refunded"]])
    return buf.getvalue().encode()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    st = documents.DocumentStore(tmp_path)
    monkeypatch.setattr(documents, "store", lambda: st)
    return st


@pytest.fixture()
def library(store):
    return {name: store.add(name, data).id for name, data in [
        ("leave_policy.pdf", LEAVE_PDF), ("falcon_status.docx", _docx()), ("q3_review.pptx", _pptx()),
        ("orders.csv", _csv())]}


def test_chunks_cover_text_and_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(60))
    spans = documents.chunk_spans(text, size=40, overlap=10)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(len(text[s:e].split()) <= 40 for s, e in spans)
    assert all(b[0] < a[1] for a, b in zip(spans, spans[1:]))            # each chunk starts inside the previous one
    assert all(text[s:e].endswith(".") for s, e in spans)                # cut at sentence ends


def test_extraction_keeps_locations(library, store):
    (pdf, docx, pptx, table), _ = store.get(list(library.values()))
    assert pdf.info.pages == 2 and [u.location for u in pdf.units] == ["p. 1", "p. 2"]
    assert [u.location for u in docx.units] == ["Timeline", "Budget"]   # the title joins the first section
    assert "Cloud hosting | 900000" in docx.units[1].text
    assert pptx.info.pages == 2 and pptx.units[1].location == "slide 2"
    assert "customer: Kabir; city: Pune; status: refunded" in table.units[1].text


def test_rejects_legacy_and_dedupes(store):
    with pytest.raises(documents.UnsupportedFile, match="save it as .docx"):
        store.add("old.doc", b"\xd0\xcf\x11\xe0")
    a, b = store.add("a.csv", _csv()), store.add("copy.csv", _csv())
    assert a.id == b.id and len(store.list()) == 1


def test_question_retrieves_the_right_passage(library):
    r = s0_retrieve.run("How many days of sick leave do I get?", list(library.values()))
    assert r.used and r.best_score >= r.threshold
    assert r.passages[0].filename == "leave_policy.pdf" and r.passages[0].location == "p. 2"
    assert "12 days of paid sick leave" in r.context and r.context.startswith("<documents>")


def test_unrelated_prompt_skips_documents(library):
    r = s0_retrieve.run("Write a haiku about autumn leaves.", list(library.values()))
    assert not r.used and r.context == "" and r.passages == []
    assert s0_retrieve.augment("x", r) == "x"
    assert not s0_retrieve.run("How many days of sick leave do I get?", list(library.values()), mode="off").used


def test_reference_to_a_file_passes_it_whole(library):
    r = s0_retrieve.run("Summarise the slides.", list(library.values()))
    assert r.used and r.explicit_reference == "the slides"
    assert {p.filename for p in r.passages} == {"q3_review.pptx"}
    assert [p.location for p in r.passages] == ["slide 1", "slide 2"]


def test_store_reloads_from_disk(library, store):
    fresh = documents.DocumentStore(store.root)
    assert {d.id for d in fresh.list()} == set(library.values())
    (doc,), _ = fresh.get([library["orders.csv"]])
    from optimus import resources
    hits = doc.search(resources.embed(["Which orders were refunded?"]), 5)
    assert "refunded" in doc.chunk_text(max(hits, key=hits.get))
    assert fresh.delete(library["orders.csv"]) and not (store.root / library["orders.csv"]).exists()


def test_api_upload_list_delete(store):
    from fastapi.testclient import TestClient

    from optimus.api import app

    client = TestClient(app)
    r = client.post("/api/v1/documents", files=[("files", ("orders.csv", _csv(), "text/csv")),
                                                ("files", ("deck.ppt", b"junk", "application/octet-stream"))])
    body = r.json()
    assert r.status_code == 200 and [d["filename"] for d in body["documents"]] == ["orders.csv"]
    assert body["errors"][0]["filename"] == "deck.ppt"
    assert [d["id"] for d in client.get("/api/v1/documents").json()] == [body["documents"][0]["id"]]
    assert client.delete(f"/api/v1/documents/{body['documents'][0]['id']}").status_code == 200
    assert client.delete("/api/v1/documents/nope").status_code == 404


def test_retrieved_passages_flow_through_the_pipeline(library):
    r = analyze_sync("Hi! Could you tell me how many days of sick leave I get? Thanks!",
                     document_ids=list(library.values()))
    assert r.s0.used and r.s1.clean_text.startswith("<documents>")
    block = next(s for s in r.s2.segments if s.kind == "tagged")
    assert block.label == "data" and "12 days of paid sick leave" in block.text
    assert r.s3.task_type.type == "qa_docs"
    assert r.s5.compression.locks_preserved and "12 days of paid sick leave" in r.s5.compression.prompt
    assert "Thanks!" not in r.s5.compression.prompt


def test_each_sub_question_gets_its_passage(library):
    r = s0_retrieve.run("What did the north region sell in Q3, and what are the risks?", list(library.values()))
    assert "what are the risks?" in r.queries
    assert {p.location for p in r.passages if p.filename == "q3_review.pptx"} == {"slide 1", "slide 2"}
