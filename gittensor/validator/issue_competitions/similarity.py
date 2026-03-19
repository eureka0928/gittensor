# The MIT License (MIT)
# Copyright © 2025 Entrius

"""Open-PR similarity detection for issue bounty reporting using shared tree-sitter change signatures.

This module intentionally keeps only grouping, pairing, attribution, and penalty
logic. The AST representation itself is shared with token scoring via
`build_change_similarity_signatures()` in `tree_sitter_scoring.py`.

Two comparison layers:
  Layer 1 — Full changed-node signatures (structural nodes + leaf text)
  Layer 2 — Structural changed-node signatures (structural node types only)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import bittensor as bt

from gittensor.classes import FileChange, Issue, MinerEvaluation, PullRequest
from gittensor.constants import (
    COPY_FILE_OVERLAP_THRESHOLD,
    COPY_FULL_THRESHOLD,
    COPY_MAX_GROUP_SIZE,
    COPY_STRUCTURAL_THRESHOLD,
)
from gittensor.validator.oss_contributions.scoring import is_valid_issue
from gittensor.validator.utils.load_weights import TokenConfig, load_token_config
from gittensor.validator.utils.tree_sitter_scoring import build_change_similarity_signatures


@dataclass
class SimilarityResult:
    similarity: float
    layer: int  # 1 = full AST, 2 = structural AST


@dataclass
class CopyDetectionResult:
    is_copy: bool
    similarity: float
    layer: int
    penalty: float  # 0.0-1.0 multiplier
    original_pr_number: Optional[int] = None
    original_repo: Optional[str] = None
    original_uid: Optional[int] = None


_TOKEN_CONFIG: Optional[TokenConfig] = None


def detect_cross_miner_copies(
    miner_evaluations: Dict[int, MinerEvaluation],
) -> Dict[Tuple[str, int], CopyDetectionResult]:
    """Detect valid issue-bounty OPEN PRs that are copies of other miners' submissions."""
    results: Dict[Tuple[str, int], CopyDetectionResult] = {}

    groups = _build_pr_groups(miner_evaluations)
    if not groups:
        return results

    bt.logging.info(f'Copy detection: {len(groups)} PR groups to analyze')

    for group_key, prs in groups.items():
        if len(prs) > COPY_MAX_GROUP_SIZE:
            prs = _cap_group_for_comparison(prs)

        for i in range(len(prs)):
            for j in range(i + 1, len(prs)):
                pr_a, pr_b = prs[i], prs[j]

                if pr_a.uid == pr_b.uid:
                    continue
                if not pr_a.file_changes or not pr_b.file_changes:
                    continue

                result = _compare_pr_pair(pr_a, pr_b)
                if result is None:
                    continue

                original, copy = _determine_originator(pr_a, pr_b)
                penalty = _calculate_copy_penalty(result.similarity, result.layer)
                copy_key = (copy.repository_full_name, copy.number)

                if copy_key in results and results[copy_key].penalty <= penalty:
                    continue

                results[copy_key] = CopyDetectionResult(
                    is_copy=True,
                    similarity=result.similarity,
                    layer=result.layer,
                    penalty=penalty,
                    original_pr_number=original.number,
                    original_repo=original.repository_full_name,
                    original_uid=original.uid,
                )

                bt.logging.warning(
                    f'Copy detected: PR #{copy.number} (uid={copy.uid}) copied from '
                    f'PR #{original.number} (uid={original.uid}) in {copy.repository_full_name} '
                    f'| layer={result.layer} similarity={result.similarity:.2f} penalty={penalty:.2f}'
                )

    bt.logging.info(f'Copy detection complete: {len(results)} copies found')
    return results


def clear_pr_file_contents(miner_evaluations: Dict[int, MinerEvaluation]) -> None:
    """Clear in-memory file contents once similarity detection is complete."""
    for evaluation in miner_evaluations.values():
        for pr in evaluation.merged_pull_requests + evaluation.open_pull_requests + evaluation.closed_pull_requests:
            pr.file_contents = None


def _build_pr_groups(
    miner_evaluations: Dict[int, MinerEvaluation],
) -> Dict[str, List[PullRequest]]:
    """Group valid issue-linked OPEN PRs for comparison by shared issue or file overlap."""
    all_prs: List[PullRequest] = []
    for evaluation in miner_evaluations.values():
        for pr in evaluation.open_pull_requests:
            if pr.file_changes and _get_valid_issues(pr):
                all_prs.append(pr)

    if not all_prs:
        return {}

    repo_prs: Dict[str, List[PullRequest]] = {}
    for pr in all_prs:
        repo_prs.setdefault(pr.repository_full_name, []).append(pr)

    groups: Dict[str, List[PullRequest]] = {}

    for repo, prs in repo_prs.items():
        issue_groups: Dict[int, List[PullRequest]] = {}
        prs_grouped_by_issue: Set[int] = set()

        for pr in prs:
            for issue in _get_valid_issues(pr):
                issue_groups.setdefault(issue.number, []).append(pr)

        for issue_num, issue_prs in issue_groups.items():
            uids = {pr.uid for pr in issue_prs}
            if len(uids) >= 2:
                groups[f'{repo}:issue:{issue_num}'] = issue_prs
                prs_grouped_by_issue.update(pr.number for pr in issue_prs)

        ungrouped = [pr for pr in prs if pr.number not in prs_grouped_by_issue]
        if len(ungrouped) >= 2:
            for i in range(len(ungrouped)):
                for j in range(i + 1, len(ungrouped)):
                    pr_a, pr_b = ungrouped[i], ungrouped[j]
                    if pr_a.uid == pr_b.uid:
                        continue
                    if _file_overlap_ratio(pr_a, pr_b) >= COPY_FILE_OVERLAP_THRESHOLD:
                        key = f'{repo}:files:{min(pr_a.number, pr_b.number)}:{max(pr_a.number, pr_b.number)}'
                        if key not in groups:
                            groups[key] = [pr_a, pr_b]

    return groups


def _get_valid_issues(pr: PullRequest) -> List[Issue]:
    return [issue for issue in (pr.issues or []) if is_valid_issue(issue, pr)]


def _file_overlap_ratio(pr_a: PullRequest, pr_b: PullRequest) -> float:
    files_a = _collect_pr_file_aliases(pr_a)
    files_b = _collect_pr_file_aliases(pr_b)
    if not files_a or not files_b:
        return 0.0
    intersection = files_a & files_b
    union = files_a | files_b
    return len(intersection) / len(union) if union else 0.0


def _collect_pr_file_aliases(pr: PullRequest) -> Set[str]:
    aliases: Set[str] = set()
    for fc in pr.file_changes or []:
        aliases.add(fc.filename)
        if fc.previous_filename:
            aliases.add(fc.previous_filename)
    return aliases


def _build_file_pairs(pr_a: PullRequest, pr_b: PullRequest) -> List[Tuple[FileChange, FileChange]]:
    """Match comparable files, including renamed paths via previous_filename."""
    if not pr_a.file_changes or not pr_b.file_changes:
        return []

    pairs: List[Tuple[FileChange, FileChange]] = []
    used_b: Set[int] = set()

    for file_a in pr_a.file_changes:
        aliases_a = {file_a.filename}
        if file_a.previous_filename:
            aliases_a.add(file_a.previous_filename)

        best_index: Optional[int] = None
        best_score = 0

        for index, file_b in enumerate(pr_b.file_changes):
            if index in used_b:
                continue
            aliases_b = {file_b.filename}
            if file_b.previous_filename:
                aliases_b.add(file_b.previous_filename)
            if not (aliases_a & aliases_b):
                continue

            score = 2 if file_a.filename == file_b.filename else 1
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            used_b.add(best_index)
            pairs.append((file_a, pr_b.file_changes[best_index]))

    return pairs


def _cap_group_for_comparison(prs: List[PullRequest]) -> List[PullRequest]:
    """Cap group, keeping the earliest PR for correct attribution."""
    if len(prs) <= COPY_MAX_GROUP_SIZE:
        return prs
    earliest_pr = min(prs, key=_origin_sort_key)
    remaining = [pr for pr in prs if pr is not earliest_pr]
    recent = sorted(remaining, key=lambda p: p.created_at, reverse=True)[: COPY_MAX_GROUP_SIZE - 1]
    return [earliest_pr, *recent]


def _origin_sort_key(pr: PullRequest) -> Tuple:
    return (pr.created_at, pr.merged_at is None, pr.merged_at or pr.created_at, pr.number)


def _compare_pr_pair(pr_a: PullRequest, pr_b: PullRequest) -> Optional[SimilarityResult]:
    """Compare two PRs using shared tree-sitter change signatures."""
    file_pairs = _build_file_pairs(pr_a, pr_b)

    if file_pairs:
        result = _compare_file_pairs(pr_a, pr_b, file_pairs)
        if result is not None:
            return result

        a_count = len(pr_a.file_changes or [])
        b_count = len(pr_b.file_changes or [])
        if len(file_pairs) >= a_count and len(file_pairs) >= b_count:
            return None

    return _compare_whole_pr(pr_a, pr_b)


def _compare_file_pairs(
    pr_a: PullRequest,
    pr_b: PullRequest,
    file_pairs: List[Tuple[FileChange, FileChange]],
) -> Optional[SimilarityResult]:
    """Compare matched file pairs using shared AST signatures weighted by changed nodes."""
    total_weight = 0.0
    weighted_full = 0.0
    weighted_structural = 0.0

    for fc_a, fc_b in file_pairs:
        full_a, struct_a, count_a = _get_file_similarity_signatures(pr_a, fc_a)
        full_b, struct_b, count_b = _get_file_similarity_signatures(pr_b, fc_b)
        weight = max(count_a, count_b)

        if weight == 0:
            continue

        weighted_full += _counter_jaccard(full_a, full_b) * weight
        weighted_structural += _counter_jaccard(struct_a, struct_b) * weight
        total_weight += weight

    if total_weight == 0:
        return None

    avg_full = weighted_full / total_weight
    avg_structural = weighted_structural / total_weight

    if avg_full >= COPY_FULL_THRESHOLD:
        return SimilarityResult(similarity=avg_full, layer=1)
    if avg_structural >= COPY_STRUCTURAL_THRESHOLD:
        return SimilarityResult(similarity=avg_structural, layer=2)

    return None


def _compare_whole_pr(pr_a: PullRequest, pr_b: PullRequest) -> Optional[SimilarityResult]:
    """Whole-PR comparison for renamed files — aggregate shared AST signatures across all files."""
    full_a, struct_a = _collect_pr_signatures(pr_a)
    full_b, struct_b = _collect_pr_signatures(pr_b)

    if not full_a and not full_b:
        return None

    full_sim = _counter_jaccard(full_a, full_b)
    structural_sim = _counter_jaccard(struct_a, struct_b)

    if full_sim >= COPY_FULL_THRESHOLD:
        return SimilarityResult(similarity=full_sim, layer=1)
    if structural_sim >= COPY_STRUCTURAL_THRESHOLD:
        return SimilarityResult(similarity=structural_sim, layer=2)

    return None


def _get_file_similarity_signatures(
    pr: PullRequest,
    file_change: FileChange,
) -> Tuple[Counter[Tuple[str, ...]], Counter[Tuple[str, ...]], int]:
    if not pr.file_contents:
        return Counter(), Counter(), 0

    file_content_pair = pr.file_contents.get(file_change.filename)
    if file_content_pair is None:
        return Counter(), Counter(), 0

    return build_change_similarity_signatures(
        file_content_pair.old_content,
        file_content_pair.new_content,
        file_change.file_extension or '',
        _get_token_config(),
    )


def _collect_pr_signatures(pr: PullRequest) -> Tuple[Counter[Tuple[str, ...]], Counter[Tuple[str, ...]]]:
    """Aggregate shared change signatures across all files in a PR."""
    full: Counter[Tuple[str, ...]] = Counter()
    structural: Counter[Tuple[str, ...]] = Counter()

    for fc in pr.file_changes or []:
        file_full, file_structural, _ = _get_file_similarity_signatures(pr, fc)
        full += file_full
        structural += file_structural

    return full, structural


def _get_token_config() -> TokenConfig:
    global _TOKEN_CONFIG
    if _TOKEN_CONFIG is None:
        _TOKEN_CONFIG = load_token_config()
    return _TOKEN_CONFIG


def _counter_jaccard(a: Counter[Tuple[str, ...]], b: Counter[Tuple[str, ...]]) -> float:
    """Multiset Jaccard: sum(min counts) / sum(max counts)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 0.0
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    return intersection / union if union > 0 else 0.0


def _determine_originator(
    pr_a: PullRequest,
    pr_b: PullRequest,
) -> Tuple[PullRequest, PullRequest]:
    """Earlier created_at wins. Tie-break: merged > open > lower PR number."""
    if pr_a.created_at != pr_b.created_at:
        return (pr_a, pr_b) if pr_a.created_at < pr_b.created_at else (pr_b, pr_a)

    if pr_a.merged_at and pr_b.merged_at:
        if pr_a.merged_at != pr_b.merged_at:
            return (pr_a, pr_b) if pr_a.merged_at < pr_b.merged_at else (pr_b, pr_a)
    elif pr_a.merged_at:
        return (pr_a, pr_b)
    elif pr_b.merged_at:
        return (pr_b, pr_a)

    return (pr_a, pr_b) if pr_a.number <= pr_b.number else (pr_b, pr_a)


def _calculate_copy_penalty(similarity: float, layer: int) -> float:
    """Penalty multiplier (0.0 = zero score, 1.0 = no penalty)."""
    if layer == 1:
        return 0.0
    elif layer == 2:
        return max(0.0, 1.0 - similarity * 1.2)
    return 1.0
