"""The two independently generated Sonnet conditions in Experiment 1."""

import csv
import hashlib
import json
from dataclasses import dataclass

from .reader import CONTROL, HERE, NOV, Prompt, extract_verdict, infer_question, json_rows


SELF_AITA = NOV / "sonnet_self_aita250_opus_sol_20261123"
TERRA_SELF = NOV / "terra_self_aita250_opus_sol_20261124"
ANNOTATIONS = HERE / "sonnet_self_aita.tsv"
LABELS = {"YTA", "NTA"}


@dataclass
class SelfFlip:
    prompt: Prompt
    target: str
    sol_response: str
    sol_raw_response: str = ""


def load_self_flips(control_pairs):
    manifest = json.loads((SELF_AITA / "experiment.json").read_text(encoding="utf-8"))
    if manifest["run"] != 1 or manifest["stance"] != "neutral":
        raise ValueError("Expected the neutral run-one Self-AITA experiment.")
    with ANNOTATIONS.open(encoding="utf-8", newline="") as source:
        annotations = list(csv.DictReader(source, delimiter="\t"))
    if len(annotations) != len({r["id"] for r in annotations}):
        raise ValueError("Self-AITA annotations must cover each annotated case once.")
    annotations = {r["id"]: r for r in annotations}

    raw = {r["pair_id"]: r for r in json_rows(SELF_AITA / "rewrites.jsonl")
           if r["custom_id"].endswith("_r1")}
    results = {}
    for judge, folder in (("opus", "judge"), ("sol", "sol")):
        rows = list(json_rows(SELF_AITA / folder / "results.jsonl"))
        scores = {(r["pair_id"], r["version"]): r for r in rows
                  if r.get("custom_id", "").endswith("_r1") or
                  (judge == "opus" and r["version"] == "original" and r["origin"] == "exact_prior_reuse")}
        if len(scores) != len(rows):
            raise ValueError(f"Unexpected or repeated {judge} records in the Self-AITA experiment.")
        results[judge] = scores

    flips = {}
    for pid in control_pairs:
        ann = annotations.get(pid, {})
        writer = raw[pid]
        original = results["opus"][pid, "original"]["prompt"]
        if not original.endswith(writer["original_story"]):
            raise ValueError(f"Original story mismatch for {pid}.")
        prefix = original.removesuffix(writer["original_story"])
        if prefix != "Output only YTA or NTA.\n":
            raise ValueError(f"Unexpected judge prefix for {pid}.")
        for judge, scores in results.items():
            og = scores[pid, "original"]
            if og["prompt"] != original or extract_verdict(og["raw_response"], og["error"]) != control_pairs[pid][f"{judge}_original"]:
                raise ValueError(f"The original input or verdict differs for {pid}, {judge}.")
        if writer["is_NA"]:
            if any(ann.get(k) for k in ("question", "speaker", "judged")) or any(
                (pid, "flip") in scores for scores in results.values()
            ):
                raise ValueError(f"N/A rewrite has annotations or verdicts for {pid}.")
            flips[pid] = SelfFlip(Prompt("", "", "", "", status="The writer returned N/A. No verdict was collected."), "", "")
            continue
        opus = results["opus"][pid, "flip"]
        sol = results["sol"][pid, "flip"]
        if opus["prompt"] != sol["prompt"] or opus["prompt"] != prefix + writer["raw_response"]:
            raise ValueError(f"The judges did not receive the same saved rewrite for {pid}.")
        if ann and (not ann["speaker"] or not ann["judged"]):
            raise ValueError(f"Missing Self-AITA roles for {pid}.")
        prompt = Prompt(opus["prompt"], ann.get("question") or infer_question(writer["raw_response"]),
                        ann.get("speaker", ""), extract_verdict(opus["raw_response"], opus["error"]),
                        raw_response=opus["raw_response"])
        flips[pid] = SelfFlip(prompt, ann.get("judged", ""), extract_verdict(sol["raw_response"], sol["error"]),
                              sol["raw_response"])
    template = manifest["rewrite_prompt_before_story"] + "[original story]" + manifest["rewrite_prompt_after_story"]
    return flips, template


def terra_self_revision():
    paths = [TERRA_SELF / name for name in ("completion_verification.json", "experiment.json", "rewrites.jsonl")]
    paths.extend(TERRA_SELF / judge / "combined_results.jsonl" for judge in ("opus", "sol"))
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


def load_terra_self_flips(control_pairs):
    marker_file = TERRA_SELF / "completion_verification.json"
    if not marker_file.exists():
        return None
    marker = json.loads(marker_file.read_text())
    protocol = json.loads((TERRA_SELF / "experiment.json").read_text())
    if not marker["passed"] or protocol["run"] != 1 or protocol["stance"] != "neutral":
        raise ValueError("Expected completed neutral run-one Terra Self-AITA results.")
    def check(path, digest):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Terra Self-AITA source changed: {path.name}")
    check(TERRA_SELF / "rewrites.jsonl", marker["rewrite_sha256"])
    check(TERRA_SELF / "summary.json", marker["summary_sha256"])
    generations = {r["pair_id"]: r for r in json_rows(TERRA_SELF / "rewrites.jsonl")}
    if set(generations) != set(control_pairs) or len(generations) != 250:
        raise ValueError("Terra Self-AITA does not match the frozen 250 stories.")
    scores = {}
    for judge in ("opus", "sol"):
        path = TERRA_SELF / judge / "combined_results.jsonl"
        check(path, marker["result_sha256"][judge])
        rows = list(json_rows(path))
        scores[judge] = {(r["pair_id"], r["version"]): r for r in rows}
        if len(rows) != len(scores[judge]) or any(r["run"] != 1 or r["stance"] != "neutral" for r in rows):
            raise ValueError("Unexpected or repeated Terra Self-AITA judgments.")
    flips = {}
    prefix = "Output only YTA or NTA.\n"
    for pid, writer in generations.items():
        for judge in ("opus", "sol"):
            row = scores[judge][pid, "original"]
            if row["prompt"] != prefix + writer["original_story"] or extract_verdict(row["raw_response"], row["error"]) != control_pairs[pid][f"{judge}_original"]:
                raise ValueError("A Terra original differs from the fixed control original.")
        if not writer["eligible"]:
            if any((pid, "flip") in scores[j] for j in scores):
                raise ValueError("An ineligible Terra rewrite was judged.")
            flips[pid] = SelfFlip(Prompt("", "", "", "", status=f"Terra generation: {writer['issue']}. No verdict was collected."), "", "")
            continue
        opus, sol = (scores[j][pid, "flip"] for j in ("opus", "sol"))
        if opus["prompt"] != sol["prompt"] or opus["prompt"] != prefix + writer["raw_response"]:
            raise ValueError("Terra Self-AITA judges received different posts.")
        flips[pid] = SelfFlip(
            Prompt(opus["prompt"], infer_question(writer["raw_response"]), "",
                   extract_verdict(opus["raw_response"], opus["error"]), raw_response=opus["raw_response"]),
            "", extract_verdict(sol["raw_response"], sol["error"]), sol["raw_response"],
        )
    template = protocol["rewrite_prompt_before_story"] + "[original story]" + protocol["rewrite_prompt_after_story"]
    return flips, template


def matched_results(control_pairs, self_flips, include_released=False, *, all_available=False, writer="sonnet"):
    """Use the same complete IDs for every displayed flip condition and judge."""
    conditions = ((writer, "Paper’s rewrite prompt"), ("self", "Self-AITA rewrite prompt"))
    if include_released:
        conditions = (("released", "Paper’s released flip"),
                      (writer, f"{writer.title()} · Paper’s prompt"),
                      ("self", f"{writer.title()} · Self-AITA prompt"))
    complete = []
    for pid, base in control_pairs.items():
        changed = self_flips[pid]
        labels = {base["opus_original"], base["sol_original"], base[f"opus_{writer}_flip"],
                  base[f"sol_{writer}_flip"], changed.prompt.verdict, changed.sol_response}
        if include_released:
            labels.update((base["opus_released_flip"], base["sol_released_flip"]))
        if labels <= LABELS:
            complete.append(pid)
    rows = []
    for source, heading in conditions:
        row = {"Flip" if include_released else "Sonnet rewrite condition": heading}
        for judge, name in (("opus", "Opus 5.5"), ("sol", "Sol")):
            count = 0
            total = 0
            for pid in control_pairs if all_available else complete:
                base, changed = control_pairs[pid], self_flips[pid]
                if source == "self":
                    flip = changed.prompt.verdict if judge == "opus" else changed.sol_response
                else:
                    flip = base[f"{judge}_{source}_flip"]
                if base[f"{judge}_original"] in LABELS and flip in LABELS:
                    total += 1
                    count += base[f"{judge}_original"] == flip == "NTA"
            row[name] = rate(count, total)
        rows.append(row)
    return rows, set(complete)


def rate(count, total):
    return f"{count / total:.1%} ({count}/{total})" if total else "— (0/0)"


def verdict_patterns(control_pairs, self_flips, case_ids, *, all_available=False):
    """Count explicit original-then-rewrite verdict patterns on the matched cohort."""
    patterns = (
        ("Sonnet · Paper’s prompt", "sonnet", "YTA"),
        ("Sonnet · Self-AITA prompt", "self", "YTA"),
        ("Sonnet · Self-AITA prompt", "self", "NTA"),
    )
    rows = []
    for heading, source, flip_label in patterns:
        row = {"Flip": heading, "Original → rewrite": f"NTA / {flip_label}"}
        for judge, name in (("opus", "Opus 5.5"), ("sol", "Sol")):
            count = 0
            total = 0
            for pid in case_ids:
                base, changed = control_pairs[pid], self_flips[pid]
                if source == "self":
                    response = changed.prompt.verdict if judge == "opus" else changed.sol_response
                else:
                    response = base[f"{judge}_sonnet_flip"]
                if base[f"{judge}_original"] in LABELS and response in LABELS:
                    total += 1
                    count += base[f"{judge}_original"] == "NTA" and response == flip_label
            row[name] = rate(count, total if all_available else len(case_ids))
        rows.append(row)
    return rows


def full_sample_results():
    paper = json.loads((CONTROL / "paired_summary.json").read_text())["by_rewrite_model_all_valid"]["sonnet"]
    sol = json.loads((NOV / "paper_control_sonnet_sol_20261123" / "summary.json").read_text())
    changed = json.loads((SELF_AITA / "summary.json").read_text())
    return [
        {"Sonnet rewrite condition": "Paper’s rewrite prompt",
         "Opus 5.5": rate(paper["NTA_NTA"], paper["valid_pairs"]),
         "Sol": rate(sol["NTA_NTA"], sol["valid_pairs"])},
        {"Sonnet rewrite condition": "Self-AITA rewrite prompt",
         "Opus 5.5": rate(changed["opus"]["NTA_NTA"], changed["opus"]["pairs_with_both_exact_labels"]),
         "Sol": rate(changed["sol"]["NTA_NTA"], changed["sol"]["pairs_with_both_exact_labels"])},
    ]
