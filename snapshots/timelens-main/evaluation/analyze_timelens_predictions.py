#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from timelens.utils import extract_time, iou


DATASET_VIDEO_DIR = {
    "activitynet-timelens": "activitynet",
    "charades-timelens": "charades",
    "qvhighlights-timelens": "qvhighlights",
}


def parse_datasets(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_predictions(path):
    merged = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                merged.update(json.loads(line))
    return merged


def first_timestamp(pred):
    source = "timestamps"
    timestamps = pred.get("timestamps")
    if not timestamps:
        source = "answers"
        timestamps = extract_time(pred.get("answers", ""))
    if not timestamps:
        return None, source, "no_timestamp"
    first = timestamps[0]
    if len(timestamps) > 1:
        status = "multiple_timestamps"
    else:
        status = "ok"
    return [float(first[0]), float(first[1])], source, status


def length_bucket(seconds, duration):
    if seconds <= 3:
        return "very_short"
    if seconds <= 8:
        return "short"
    if seconds <= 20:
        return "medium"
    if duration and seconds / duration >= 0.4:
        return "most_of_video"
    return "long"


def position_bucket(center, duration):
    if not duration:
        return "unknown"
    ratio = center / duration
    if ratio < 0.33:
        return "early"
    if ratio < 0.66:
        return "middle"
    return "late"


def add_tags(row):
    tags = []
    if row["timestamp_status"] != "ok":
        tags.append(row["timestamp_status"])
    if row["pred_start"] is None or row["pred_start"] >= row["pred_end"]:
        tags.append("invalid_or_empty_prediction")
        row["tags"] = ";".join(tags)
        return

    gt_len = max(row["gt_end"] - row["gt_start"], 1e-6)
    pred_len = max(row["pred_end"] - row["pred_start"], 1e-6)
    duration = row["duration"]
    center_shift = ((row["pred_start"] + row["pred_end"]) - (row["gt_start"] + row["gt_end"])) / 2

    if row["iou"] < 0.3:
        tags.append("miss")
    elif row["iou"] >= 0.7:
        tags.append("high_iou")
    if pred_len / gt_len > 2.0:
        tags.append("over_extended")
    if pred_len / gt_len < 0.5:
        tags.append("under_extended")
    if duration and abs(center_shift) > max(5.0, 0.1 * duration):
        tags.append("large_center_shift")
    if center_shift < -max(3.0, 0.1 * gt_len):
        tags.append("early_prediction")
    if center_shift > max(3.0, 0.1 * gt_len):
        tags.append("late_prediction")
    if row["pred_start"] <= 1.0:
        tags.append("starts_at_video_beginning")
    if duration and row["pred_end"] >= duration - 1.0:
        tags.append("ends_at_video_end")
    if gt_len <= 3:
        tags.append("tiny_ground_truth")
    if duration and gt_len / duration >= 0.4:
        tags.append("broad_ground_truth")
    row["tags"] = ";".join(tags)


def make_rows(pred_run_dir, data_root, datasets, baseline_run_dir=None):
    rows = []
    baseline = {}
    if baseline_run_dir:
        for dataset in datasets:
            path = baseline_run_dir / f"{dataset}.jsonl"
            if path.exists():
                baseline[dataset] = read_predictions(path)

    for dataset in datasets:
        pred_path = pred_run_dir / f"{dataset}.jsonl"
        if not pred_path.exists():
            print(f"warning: missing prediction file {pred_path}")
            continue
        predictions = read_predictions(pred_path)
        video_dir = DATASET_VIDEO_DIR[dataset]

        for key, pred in predictions.items():
            video_name, query, gt_span_raw = key.split(">>>")
            gt_span = ast.literal_eval(gt_span_raw)
            if isinstance(gt_span[0], (list, tuple)):
                gt_span = gt_span[0]
            gt_start, gt_end = float(gt_span[0]), float(gt_span[1])
            duration = float(pred.get("duration", 0) or 0)
            timestamp, source, status = first_timestamp(pred)
            if timestamp is None:
                pred_start = pred_end = None
                sample_iou = 0.0
            else:
                pred_start, pred_end = timestamp
                sample_iou = iou([gt_start, gt_end], [pred_start, pred_end])

            base_iou = None
            base_answer = ""
            if dataset in baseline and key in baseline[dataset]:
                base_pred = baseline[dataset][key]
                base_timestamp, _, _ = first_timestamp(base_pred)
                if base_timestamp is not None:
                    base_iou = iou([gt_start, gt_end], base_timestamp)
                else:
                    base_iou = 0.0
                base_answer = base_pred.get("answers", "")

            row = {
                "dataset": dataset,
                "video": video_name,
                "video_path": str(data_root / "videos" / video_dir / video_name),
                "query": query,
                "duration": duration,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "gt_len": gt_end - gt_start,
                "gt_len_bucket": length_bucket(gt_end - gt_start, duration),
                "gt_position": position_bucket((gt_start + gt_end) / 2, duration),
                "pred_start": pred_start,
                "pred_end": pred_end,
                "pred_len": None if pred_start is None else pred_end - pred_start,
                "iou": sample_iou,
                "baseline_iou": base_iou,
                "answer": pred.get("answers", ""),
                "baseline_answer": base_answer,
                "timestamp_source": source,
                "timestamp_status": status,
                "tags": "",
            }
            add_tags(row)
            rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def summarize(rows):
    summary = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        items = [row for row in rows if row["dataset"] == dataset]
        count = len(items)
        summary[dataset] = {
            "count": count,
            "r1@0.3": 100 * sum(row["iou"] >= 0.3 for row in items) / count,
            "r1@0.5": 100 * sum(row["iou"] >= 0.5 for row in items) / count,
            "r1@0.7": 100 * sum(row["iou"] >= 0.7 for row in items) / count,
            "miou": 100 * sum(row["iou"] for row in items) / count,
            "invalid": sum("invalid_or_empty_prediction" in row["tags"] for row in items),
        }
    return summary


def bucket_summary(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row[field])].append(row)
    records = []
    for (dataset, bucket), items in sorted(grouped.items()):
        records.append(
            {
                "dataset": dataset,
                field: bucket,
                "count": len(items),
                "r1@0.3": 100 * sum(row["iou"] >= 0.3 for row in items) / len(items),
                "r1@0.7": 100 * sum(row["iou"] >= 0.7 for row in items) / len(items),
                "miou": 100 * sum(row["iou"] for row in items) / len(items),
            }
        )
    return records


def tag_summary(rows):
    records = []
    for dataset in sorted({row["dataset"] for row in rows}):
        counts = Counter()
        for row in rows:
            if row["dataset"] != dataset:
                continue
            for tag in row["tags"].split(";"):
                if tag:
                    counts[tag] += 1
        for tag, count in counts.most_common():
            records.append({"dataset": dataset, "tag": tag, "count": count})
    return records


def select_examples(rows, count):
    failures = sorted(
        [row for row in rows if row["iou"] < 0.3],
        key=lambda row: (
            -(row["baseline_iou"] or 0),
            "invalid_or_empty_prediction" in row["tags"],
            -row["duration"],
        ),
    )[:count]
    successes = sorted(
        [row for row in rows if row["iou"] >= 0.7],
        key=lambda row: (-(row["baseline_iou"] is not None and row["baseline_iou"] < 0.3), -row["iou"]),
    )[:count]
    return failures, successes


def distance(a, b):
    return (
        abs(math.log1p(a["duration"]) - math.log1p(b["duration"]))
        + abs(math.log1p(max(a["gt_len"], 0)) - math.log1p(max(b["gt_len"], 0)))
        + abs(len(a["query"]) - len(b["query"])) / 100
    )


def matched_examples(failures, successes):
    matches = []
    for fail in failures:
        candidates = [row for row in successes if row["dataset"] == fail["dataset"]]
        if not candidates:
            continue
        good = min(candidates, key=lambda row: distance(fail, row))
        matches.append(
            {
                "dataset": fail["dataset"],
                "failure_video": fail["video"],
                "failure_video_path": fail["video_path"],
                "failure_query": fail["query"],
                "failure_gt": f"{fail['gt_start']:.2f}-{fail['gt_end']:.2f}",
                "failure_pred": f"{fail['pred_start']}-{fail['pred_end']}",
                "failure_iou": fail["iou"],
                "failure_baseline_iou": fail["baseline_iou"],
                "success_video": good["video"],
                "success_video_path": good["video_path"],
                "success_query": good["query"],
                "success_gt": f"{good['gt_start']:.2f}-{good['gt_end']:.2f}",
                "success_pred": f"{good['pred_start']}-{good['pred_end']}",
                "success_iou": good["iou"],
                "success_baseline_iou": good["baseline_iou"],
            }
        )
    return matches


def copy_videos(rows, output_dir, limit):
    if limit <= 0:
        return []
    copied = []
    for row in rows[:limit]:
        src = Path(row["video_path"])
        if src.exists():
            dst = output_dir / "videos" / row["dataset"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def make_relative(path, base):
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def copy_pair_videos(matches, output_dir, limit):
    copied = {}
    for match in matches[:limit]:
        for key in ("failure_video_path", "success_video_path"):
            src = Path(match[key])
            if not src.exists():
                continue
            dst = output_dir / "videos" / match["dataset"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied[str(src)] = str(dst)
    return copied


def write_demo_html(path, matches, copied_videos, max_pairs):
    rows = []
    for index, match in enumerate(matches[:max_pairs], start=1):
        failure_src = copied_videos.get(match["failure_video_path"], match["failure_video_path"])
        success_src = copied_videos.get(match["success_video_path"], match["success_video_path"])
        failure_rel = make_relative(failure_src, path.parent)
        success_rel = make_relative(success_src, path.parent)
        rows.append(
            f"""
      <section class="pair">
        <h2>{index}. {match['dataset']}</h2>
        <div class="grid">
          <article>
            <h3>Failure candidate</h3>
            <video controls preload="metadata" src="{failure_rel}"></video>
            <p><b>Query:</b> {match['failure_query']}</p>
            <p><b>GT:</b> {match['failure_gt']} seconds</p>
            <p><b>Prediction:</b> {match['failure_pred']} seconds</p>
            <p><b>IoU:</b> {match['failure_iou']:.3f}</p>
          </article>
          <article>
            <h3>Comparable success</h3>
            <video controls preload="metadata" src="{success_rel}"></video>
            <p><b>Query:</b> {match['success_query']}</p>
            <p><b>GT:</b> {match['success_gt']} seconds</p>
            <p><b>Prediction:</b> {match['success_pred']} seconds</p>
            <p><b>IoU:</b> {match['success_iou']:.3f}</p>
          </article>
        </div>
      </section>
"""
        )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>TimeLens-8B Good-vs-Bad Video Demo</title>
    <style>
      body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #111; }}
      h1 {{ margin-bottom: 8px; }}
      .note {{ max-width: 900px; color: #444; }}
      .pair {{ border-top: 1px solid #ddd; padding-top: 24px; margin-top: 28px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }}
      video {{ width: 100%; max-height: 420px; background: #000; }}
      article {{ min-width: 0; }}
      p {{ margin: 8px 0; }}
      @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <h1>TimeLens-8B Good-vs-Bad Video Demo</h1>
    <p class="note">Each pair shows an actual low-IoU TimeLens-8B prediction next to a comparable high-IoU prediction from the same dataset. Use these examples to inspect visible temporal ambiguity, event boundary precision, repeated actions, and short-event localization.</p>
    {''.join(rows)}
  </body>
</html>
"""
    path.write_text(html)


def write_report(path, summary, tag_rows, failures, successes, matches):
    lines = ["# TimeLens-8B Reproduction Analysis", ""]
    lines.append("## Metrics")
    lines.append("| Dataset | Count | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | Invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset, item in summary.items():
        lines.append(
            f"| {dataset} | {item['count']} | {item['r1@0.3']:.2f} | {item['r1@0.5']:.2f} | "
            f"{item['r1@0.7']:.2f} | {item['miou']:.2f} | {item['invalid']} |"
        )
    lines.append("")
    lines.append("## Frequent Failure Tags")
    lines.append("| Dataset | Tag | Count |")
    lines.append("|---|---|---:|")
    for row in tag_rows[:40]:
        lines.append(f"| {row['dataset']} | {row['tag']} | {row['count']} |")
    lines.append("")
    lines.append("## Candidate Failure Videos")
    lines.append("| Dataset | Video | Query | GT | Pred | IoU | Baseline IoU | Tags |")
    lines.append("|---|---|---|---|---|---:|---:|---|")
    for row in failures[:12]:
        baseline = "" if row["baseline_iou"] is None else f"{row['baseline_iou']:.3f}"
        lines.append(
            f"| {row['dataset']} | `{row['video_path']}` | {row['query']} | "
            f"{row['gt_start']:.2f}-{row['gt_end']:.2f} | {row['pred_start']}-{row['pred_end']} | "
            f"{row['iou']:.3f} | {baseline} | {row['tags']} |"
        )
    lines.append("")
    lines.append("## Matched Good-vs-Bad Pairs")
    lines.append("| Dataset | Failure Video | Failure Query | Failure IoU | Comparable Success Video | Success Query | Success IoU |")
    lines.append("|---|---|---|---:|---|---|---:|")
    for row in matches[:12]:
        lines.append(
            f"| {row['dataset']} | `{row['failure_video_path']}` | {row['failure_query']} | "
            f"{row['failure_iou']:.3f} | `{row['success_video_path']}` | {row['success_query']} | "
            f"{row['success_iou']:.3f} |"
        )
    lines.append("")
    lines.append("## Candidate Success Videos")
    lines.append("| Dataset | Video | Query | GT | Pred | IoU | Baseline IoU |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for row in successes[:12]:
        baseline = "" if row["baseline_iou"] is None else f"{row['baseline_iou']:.3f}"
        lines.append(
            f"| {row['dataset']} | `{row['video_path']}` | {row['query']} | "
            f"{row['gt_start']:.2f}-{row['gt_end']:.2f} | {row['pred_start']}-{row['pred_end']} | "
            f"{row['iou']:.3f} | {baseline} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--datasets", default="charades-timelens,activitynet-timelens,qvhighlights-timelens")
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--example-count", type=int, default=50)
    parser.add_argument("--copy-video-examples", type=int, default=0)
    parser.add_argument("--demo-pairs", type=int, default=12)
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    output_dir = args.output_dir or args.pred_run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = make_rows(args.pred_run_dir, args.data_root, datasets, args.baseline_run_dir)
    if not rows:
        raise SystemExit("No prediction rows found.")

    fields = [
        "dataset",
        "video",
        "video_path",
        "query",
        "duration",
        "gt_start",
        "gt_end",
        "gt_len",
        "gt_len_bucket",
        "gt_position",
        "pred_start",
        "pred_end",
        "pred_len",
        "iou",
        "baseline_iou",
        "timestamp_status",
        "tags",
        "answer",
        "baseline_answer",
    ]
    write_csv(output_dir / "per_sample.csv", rows, fields)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    bucket_fields = ["dataset", "gt_len_bucket", "count", "r1@0.3", "r1@0.7", "miou"]
    write_csv(output_dir / "by_gt_length.csv", bucket_summary(rows, "gt_len_bucket"), bucket_fields)
    position_fields = ["dataset", "gt_position", "count", "r1@0.3", "r1@0.7", "miou"]
    write_csv(output_dir / "by_gt_position.csv", bucket_summary(rows, "gt_position"), position_fields)
    tag_rows = tag_summary(rows)
    write_csv(output_dir / "tag_summary.csv", tag_rows, ["dataset", "tag", "count"])

    failures, successes = select_examples(rows, args.example_count)
    write_csv(output_dir / "candidate_failures.csv", failures, fields)
    write_csv(output_dir / "candidate_successes.csv", successes, fields)
    matches = matched_examples(failures, successes)
    write_csv(
        output_dir / "matched_good_bad_pairs.csv",
        matches,
        [
            "dataset",
            "failure_video",
            "failure_video_path",
            "failure_query",
            "failure_gt",
            "failure_pred",
            "failure_iou",
            "failure_baseline_iou",
            "success_video",
            "success_video_path",
            "success_query",
            "success_gt",
            "success_pred",
            "success_iou",
            "success_baseline_iou",
        ],
    )
    copied = copy_videos(failures + successes, output_dir, args.copy_video_examples)
    copied_pairs = copy_pair_videos(matches, output_dir, args.demo_pairs)
    if copied:
        (output_dir / "copied_videos.txt").write_text("\n".join(copied) + "\n")
    if copied_pairs:
        write_demo_html(output_dir / "demo.html", matches, copied_pairs, args.demo_pairs)
    write_report(output_dir / "report.md", summary, tag_rows, failures, successes, matches)
    print(f"analysis_dir={output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
