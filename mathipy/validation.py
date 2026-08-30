"""Sampling, coding sheets, and rater agreement for validating extraction output.

Closes the loop between automated extraction and human verification: draw a
stratified sample, emit a coding sheet or a Label Studio task file, then score
agreement between two raters. Depends only on the standard library.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mathipy.utils import compute_interrater_reliability
from mathipy.visual import visual_functions, visual_model_definitions

ocr_rubric = {
    "question_text_complete": ["complete", "partial", "missing"],
    "math_expressions_correct": ["correct", "partial", "incorrect", "none_present"],
    "answer_choices_complete": ["complete", "partial", "missing", "not_applicable"],
    "hallucinated_content": ["yes", "no"],
}

visual_rubric = {
    "primary_type": sorted(visual_model_definitions),
    "instructional_function": list(visual_functions),
    "classification_disputed": ["yes", "no"],
}


def stratum_label(record: dict[str, Any], keys: Sequence[str]) -> str:
    """Join the values of ``keys`` into a single stratum label."""
    return " | ".join(str(record.get(k, "unrecorded")) for k in keys)


def stratified_sample(
    records: Sequence[dict[str, Any]],
    strata: Sequence[str] | Callable[[dict[str, Any]], str],
    n: int = 120,
    seed: int = 0,
    min_per_stratum: int = 5,
) -> list[dict[str, Any]]:
    """Draw a sample allocated across strata, with a floor for rare strata.

    Args:
        records: Rows to sample from.
        strata: Field names to stratify on, or a callable returning a label.
        n: Target sample size before the per-stratum floor is applied.
        seed: Seed for the shuffle, so a draw is reproducible.
        min_per_stratum: Smallest number taken from any non-empty stratum.

    Returns:
        Selected rows, each with a ``stratum`` key added.
    """
    if not records:
        return []

    label = strata if callable(strata) else (lambda r: stratum_label(r, strata))
    tagged = [dict(r, stratum=label(r)) for r in records]
    counts = Counter(r["stratum"] for r in tagged)

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for stratum, size in counts.items():
        take = max(min_per_stratum, round(n * size / len(tagged)))
        rows = [r for r in tagged if r["stratum"] == stratum]
        rng.shuffle(rows)
        picked.extend(rows[:min(take, len(rows))])
    return picked


def coding_sheet(
    records: Sequence[dict[str, Any]],
    rubric: dict[str, list[str]],
    carry: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return rows carrying context fields plus one blank column per rubric field."""
    keep = list(carry) if carry else list(records[0]) if records else []
    return [
        {**{k: r.get(k, "") for k in keep}, **{f: "" for f in rubric}}
        for r in records
    ]


def write_coding_sheets(
    records: Sequence[dict[str, Any]],
    rubric: dict[str, list[str]],
    out_dir: str | Path,
    raters: Sequence[str] = ("a", "b"),
    carry: Sequence[str] | None = None,
) -> list[Path]:
    """Write one blank coding sheet per rater, plus the rubric as a text file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = coding_sheet(records, rubric, carry)
    if not rows:
        raise ValueError("no records to write")

    written = []
    for rater in raters:
        path = out / f"coding-sheet-{rater}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)

    guide = "\n".join(f"{field}: {' | '.join(options)}" for field, options in rubric.items())
    (out / "rubric.txt").write_text(guide + "\n", encoding="utf-8")
    return written


def label_studio_tasks(
    records: Sequence[dict[str, Any]],
    rubric: dict[str, list[str]],
    image_key: str = "image",
    text_key: str = "text",
    id_key: str = "item_id",
) -> list[dict[str, Any]]:
    """Convert sampled records into Label Studio task dictionaries."""
    tasks = []
    for r in records:
        data = {id_key: r.get(id_key, ""), "text": r.get(text_key, "")}
        if r.get(image_key):
            data["image"] = r[image_key]
        for extra in ("stratum", "provider", "model", "base_url", "extracted_at"):
            if r.get(extra):
                data[extra] = r[extra]
        tasks.append({"data": data, "meta": {"rubric": list(rubric)}})
    return tasks


def label_studio_config(
    rubric: dict[str, list[str]],
    image_key: str = "image",
    text_key: str = "text",
) -> str:
    """Return a Label Studio labeling-interface XML config for the rubric."""
    blocks = [
        f'  <Image name="img" value="${image_key}"/>',
        f'  <Text name="extracted" value="${text_key}"/>',
    ]
    for field, options in rubric.items():
        choices = "".join(f'\n      <Choice value="{o}"/>' for o in options)
        blocks.append(
            f'  <Header value="{field}"/>\n'
            f'  <Choices name="{field}" toName="img" choice="single" showInLine="true">'
            f"{choices}\n  </Choices>"
        )
    return "<View>\n" + "\n".join(blocks) + "\n</View>\n"


def write_label_studio(
    records: Sequence[dict[str, Any]],
    rubric: dict[str, list[str]],
    out_dir: str | Path,
    image_key: str = "image",
    text_key: str = "text",
) -> tuple[Path, Path]:
    """Write a Label Studio task file and its labeling config."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks_path = out / "label-studio-tasks.json"
    config_path = out / "label-studio-config.xml"
    tasks_path.write_text(
        json.dumps(label_studio_tasks(records, rubric, image_key, text_key), indent=2),
        encoding="utf-8",
    )
    config_path.write_text(
        label_studio_config(rubric, image_key, text_key), encoding="utf-8"
    )
    return tasks_path, config_path


def score_agreement(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    fields: Sequence[str],
    id_key: str = "item_id",
) -> dict[str, dict[str, Any]]:
    """Compute agreement and Cohen's kappa per rubric field over matched rows."""
    by_id_b = {r.get(id_key): r for r in rows_b}
    matched = [(a, by_id_b[a.get(id_key)]) for a in rows_a if a.get(id_key) in by_id_b]

    scored = {}
    for field in fields:
        pairs = [
            (a.get(field), b.get(field))
            for a, b in matched
            if str(a.get(field, "")).strip() and str(b.get(field, "")).strip()
        ]
        if not pairs:
            scored[field] = {"agreement": 0.0, "kappa": 0.0, "n": 0}
            continue
        first, second = zip(*pairs)
        scored[field] = compute_interrater_reliability(first, second)
    return scored


def disagreements(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    fields: Sequence[str],
    id_key: str = "item_id",
) -> list[dict[str, Any]]:
    """Return rows where the two raters differ on at least one rubric field."""
    by_id_b = {r.get(id_key): r for r in rows_b}
    out = []
    for a in rows_a:
        b = by_id_b.get(a.get(id_key))
        if b is None:
            continue
        differing = [f for f in fields if a.get(f) != b.get(f)]
        if differing:
            out.append({
                id_key: a.get(id_key),
                "fields": ";".join(differing),
                **{f"{f}_a": a.get(f) for f in differing},
                **{f"{f}_b": b.get(f) for f in differing},
            })
    return out




_app_template = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root { color-scheme: light dark; --line: #8884; }
 body { font: 16px/1.6 system-ui, sans-serif; max-width: 60rem; margin: 0 auto;
        padding: 1rem 1.2rem 4rem; }
 header { position: sticky; top: 0; z-index: 5; background: Canvas;
          border-bottom: 1px solid var(--line); padding: .8rem 0; margin-bottom: 1rem; }
 .row { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
 .bar { height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; margin: .5rem 0; }
 .bar > div { height: 100%; background: #3b82f6; width: 0; transition: width .2s; }
 .card { border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1.05rem;
         margin-bottom: .7rem; }
 .card.active { border-color: #3b82f6; }
 .card.done .state { color: #16a34a; }
 .meta { display: flex; justify-content: space-between; font-size: .78rem;
         opacity: .75; margin-bottom: .5rem; }
 .body { font-size: 1rem; margin-bottom: .7rem; white-space: pre-wrap; }
 img { max-width: 100%; border-radius: 6px; margin-top: .5rem; }
 select, input, button { font: inherit; }
 select { width: 100%; padding: .35rem; border-radius: 6px; }
 .field { margin-bottom: .45rem; }
 .field label { display: block; font-size: .78rem; opacity: .75; margin-bottom: .15rem; }
 .notes { width: 100%; padding: .35rem; border-radius: 6px; margin-top: .4rem; }
 button { padding: .4rem .8rem; border-radius: 6px; border: 1px solid #8886;
          background: transparent; cursor: pointer; }
 button.on { background: #3b82f6; color: #fff; border-color: #3b82f6; }
 #search { flex: 1; min-width: 10rem; padding: .4rem; border-radius: 6px;
           border: 1px solid #8886; }
 .hidden { display: none; }
 .rules { font-size: .8rem; opacity: .7; margin-top: .3rem; }
</style>
<header>
  <div class="row">
    <strong>__TITLE__</strong>
    <span style="margin-left:auto; font-size:.85rem; opacity:.75" id="who"></span>
  </div>
  <div class="bar"><div id="prog"></div></div>
  <div class="row">
    <span style="font-size:.85rem; opacity:.75" id="count"></span>
    <input id="search" placeholder="Search">
    <button id="f-todo">To do</button>
    <button id="f-all" class="on">All</button>
    <button id="download">Download</button>
  </div>
</header>
<main id="app"></main>
<script>
const records = __RECORDS__, rubric = __RUBRIC__, carry = __CARRY__;
const rules = __RULES__, title = __TITLE_JSON__, imageBase = __IMAGE_BASE__;
const fields = Object.keys(rubric);
const rater = new URLSearchParams(location.search).get("rater") ||
              prompt("Coder name or initials") || "unnamed";
const store = "coding:" + title + ":" + rater;
let answers = JSON.parse(localStorage.getItem(store) || "{}");
let onlyTodo = false, query = "";

document.getElementById("who").textContent = rater;

const esc = s => String(s).replace(/[<>&]/g, c => ({"<": "&lt;", ">": "&gt;", "&": "&amp;"}[c]));
const pretty = s => { const t = String(s).replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };
const save = () => localStorage.setItem(store, JSON.stringify(answers));
const isDone = r => Boolean((answers[String(r.__id)] || {})[fields[0]]);

function build() {
  const app = document.getElementById("app");
  app.innerHTML = "";
  for (const rec of records) {
    const key = String(rec.__id);
    answers[key] = answers[key] || {};
    const card = document.createElement("section");
    card.className = "card" + (isDone(rec) ? " done" : "");
    card.dataset.id = key;

    const shown = carry.filter(f => f !== "__id" && rec[f] !== undefined && rec[f] !== "");
    let html = '<div class="meta"><span>' + esc(rec.__id) + '</span>' +
               '<span class="state">' + (isDone(rec) ? "coded" : "not yet coded") + '</span></div>';
    html += shown.map(f => '<div class="body">' + esc(rec[f]) + '</div>').join("");
    if (imageBase && rec.__images) {
      html += rec.__images.split(";").filter(s => s.trim())
        .map(s => '<img loading="lazy" src="' + imageBase + s.trim() + '" alt="">').join("");
    }
    for (const f of fields) {
      html += '<div class="field"><label>' + esc(f) + '</label><select data-f="' + esc(f) + '">' +
        '<option value="">Choose</option>' +
        rubric[f].map(o => '<option value="' + esc(o) + '"' +
          (answers[key][f] === o ? " selected" : "") + '>' + esc(pretty(o)) + '</option>').join("") +
        '</select>' + (rules[f] ? '<div class="rules">' + esc(rules[f]) + '</div>' : '') + '</div>';
    }
    html += '<input class="notes" placeholder="Notes" value="' + esc(answers[key].notes || "") + '">';
    card.innerHTML = html;

    card.querySelectorAll("select").forEach(sel => sel.onchange = () => {
      answers[key][sel.dataset.f] = sel.value; save(); refresh();
    });
    card.querySelector(".notes").oninput = e => { answers[key].notes = e.target.value; save(); };
    card.addEventListener("focusin", () => {
      document.querySelectorAll(".card.active").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
    });
    app.appendChild(card);
  }
  refresh();
}

function refresh() {
  let visible = 0;
  for (const rec of records) {
    const card = document.querySelector('.card[data-id="' + CSS.escape(String(rec.__id)) + '"]');
    if (!card) continue;
    const done = isDone(rec);
    card.classList.toggle("done", done);
    card.querySelector(".state").textContent = done ? "coded" : "not yet coded";
    const text = carry.map(f => rec[f]).join(" ").toLowerCase();
    const show = (!onlyTodo || !done) && (!query || text.includes(query));
    card.classList.toggle("hidden", !show);
    if (show) visible++;
  }
  const done = records.filter(isDone).length;
  document.getElementById("count").textContent =
    done + " of " + records.length + " coded" +
    (visible < records.length ? " (" + visible + " shown)" : "");
  document.getElementById("prog").style.width = (100 * done / records.length) + "%";
}

document.getElementById("search").oninput = e => { query = e.target.value.toLowerCase().trim(); refresh(); };
document.getElementById("f-todo").onclick = () => {
  onlyTodo = true; document.getElementById("f-todo").classList.add("on");
  document.getElementById("f-all").classList.remove("on"); refresh();
};
document.getElementById("f-all").onclick = () => {
  onlyTodo = false; document.getElementById("f-all").classList.add("on");
  document.getElementById("f-todo").classList.remove("on"); refresh();
};
document.getElementById("download").onclick = () => {
  const cols = ["item"].concat(fields).concat(["notes"]);
  const q = v => '"' + String(v === undefined ? "" : v).replace(/"/g, '""') + '"';
  const lines = [cols.join(",")];
  for (const rec of records) {
    const a = answers[String(rec.__id)] || {};
    const row = [rec.__id].concat(fields.map(f => a[f] || "")).concat([a.notes || ""]);
    lines.push(row.map(q).join(","));
  }
  const url = URL.createObjectURL(new Blob([lines.join("\\n")], {type: "text/csv"}));
  const link = document.createElement("a");
  link.href = url; link.download = "coding-" + rater + ".csv"; link.click();
};
build();
</script>
"""


def write_coding_app(
    records: Sequence[dict[str, Any]],
    rubric: dict[str, list[str]],
    out_dir: str | Path,
    title: str = "Coding",
    carry: Sequence[str] | None = None,
    id_field: str = "item",
    image_field: str | None = None,
    image_base: str = "images/",
    rules: dict[str, str] | None = None,
) -> Path:
    """Write one offline page listing every item, for a coder to fill in and export.

    The whole task sits on one page rather than advancing one item at a time, because
    coding consistently depends on being able to scan neighbours, compare, and revise
    an earlier decision. Nothing leaves the machine the page is opened on, progress
    survives a closed tab, and the exported columns match ``score_agreement``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = list(carry) if carry else [k for k in (records[0] if records else {})]

    prepared = []
    for row in records:
        entry = {k: row.get(k, "") for k in keep}
        entry["__id"] = row.get(id_field, "")
        if image_field:
            entry["__images"] = row.get(image_field, "")
        prepared.append(entry)

    page = (_app_template
            .replace("__RECORDS__", json.dumps(prepared))
            .replace("__RUBRIC__", json.dumps({k: list(v) for k, v in rubric.items()}))
            .replace("__CARRY__", json.dumps(keep))
            .replace("__RULES__", json.dumps(rules or {}))
            .replace("__TITLE_JSON__", json.dumps(title))
            .replace("__IMAGE_BASE__", json.dumps(image_base if image_field else ""))
            .replace("__TITLE__", title))

    path = out / "coding-app.html"
    path.write_text(page, encoding="utf-8")
    return path
