"""Read the separately collected Yes/No response-format control."""

import hashlib
import json

from .reader import NOV, POSTS_ONLY, json_rows


YES_NO = NOV / "three_versions_yes_no_20261123"


def yes_no_revision():
    paths = [YES_NO / "completion_verification.json", YES_NO / "summary.json"]
    paths.extend(YES_NO / judge / "results.jsonl" for judge in ("opus", "sol"))
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in paths)


def load_yes_no():
    marker_path = YES_NO / "completion_verification.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text())
    protocol = json.loads((YES_NO / "experiment.json").read_text())
    if not marker["passed"] or protocol["run"] != 1 or protocol["stance"] != "neutral":
        raise ValueError("Expected verified neutral run-one Yes/No inputs.")
    summary_path = YES_NO / "summary.json"
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != marker["summary_sha256"]:
        raise ValueError("The Yes/No summary changed after verification.")
    results = {}
    for judge in marker["judges"]:
        path = YES_NO / judge / "results.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != marker["result_sha256"][judge]:
            raise ValueError("The Yes/No results changed after verification.")
        results[judge] = list(json_rows(path))
        source = {row["custom_id"]: row for row in json_rows(POSTS_ONLY / judge / "results.jsonl")}
        for row in results[judge]:
            previous = source[row["source_custom_id"]]
            if (row["run"] != 1 or row["stance"] != "neutral" or row["judge"] != judge
                or row["story"] != previous["story"] or row["question"] != previous["question"]
                or row["prompt"] != protocol["prefix"] + previous["story"]):
                raise ValueError("A Yes/No input differs from the saved post-only story or question.")
    return {"results": results, "summary": json.loads(summary_path.read_text()), "prefix": protocol["prefix"]}


def format_comparison(rows):
    table = []
    for row in rows:
        def rate(count, n):
            return f"{count / n:.1%} ({count}/{n})" if n else "Not scored"
        table.append({
            "Rewrite model": row["writer"].title(),
            "Judge": "Opus 5.5" if row["judge"] == "opus" else "Sol",
            "V1 / V2 · No/Yes": rate(row["after_v1_v2_No_Yes"], row["after_dilemmas"]),
            "V1 / V3 · No/No": rate(row["after_v1_v3_No_No"], row["after_dilemmas"]),
        })
    return table


def comparison_rows(yes_no, posts_only, expansion, matched_ids=None):
    """Count actual completed triplets, including the full Terra expansion."""
    output = []
    def triplets(source, judge, writer, labels):
        grouped = {}
        for row in source["results"].get(judge, []):
            if row["writer"] != writer or (matched_ids is not None and row["pair_id"] not in matched_ids):
                continue
            grouped.setdefault(row["pair_id"], {})[row["version"]] = row["label"]
        return {pid: tuple(values.get(v, "") for v in (1, 2, 3)) for pid, values in grouped.items()
                if all(values.get(v) in labels for v in (1, 2, 3))}
    for writer in ("sonnet", "terra"):
        after_source = expansion if writer == "terra" and matched_ids is None and expansion else yes_no
        for judge in ("opus", "sol"):
            before = triplets(posts_only, judge, writer, {"NTA", "YTA"})
            after = triplets(after_source, judge, writer, {"Yes", "No"})
            if matched_ids is not None:
                common = before.keys() & after.keys()
                before = {pid: before[pid] for pid in common}
                after = {pid: after[pid] for pid in common}
            output.append({
                "writer": writer, "judge": judge, "before_dilemmas": len(before), "after_dilemmas": len(after),
                "before_v1_v2_NTA_YTA": sum(v[:2] == ("NTA", "YTA") for v in before.values()),
                "before_v1_v3_NTA_NTA": sum(v[0] == v[2] == "NTA" for v in before.values()),
                "after_v1_v2_No_No": sum(v[:2] == ("No", "No") for v in after.values()),
                "after_v1_v2_No_Yes": sum(v[:2] == ("No", "Yes") for v in after.values()),
                "after_v1_v3_No_No": sum(v[0] == v[2] == "No" for v in after.values()),
                "after_v1_v3_No_Yes": sum(v[0] == "No" and v[2] == "Yes" for v in after.values()),
            })
    return output


def load_three_prompt():
    source = json.loads((NOV / "three_versions_sonnet250_20261122/experiment.json").read_text())
    expanded = json.loads((NOV / "terra500_three_versions_yes_no_20261123/experiment.json").read_text())
    if source["prompt_before_story"] != expanded["rewrite_prompt_before_story"]:
        raise ValueError("The three-version writers used different rewrite prompts.")
    return source["prompt_before_story"] + "[original story]"
