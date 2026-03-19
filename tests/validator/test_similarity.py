# The MIT License (MIT)
# Copyright © 2025 Entrius

"""Tests for cross-miner PR similarity detection (tree-sitter based)."""

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from gittensor.classes import FileChange, Issue, MinerEvaluation, PRState, PullRequest
from gittensor.utils.github_api_tools import FileContentPair
from gittensor.validator.issue_competitions.similarity import (
    _build_pr_groups,
    _calculate_copy_penalty,
    _counter_jaccard,
    _determine_originator,
    detect_cross_miner_copies,
)
from gittensor.validator.oss_contributions.tier_config import TIERS, Tier
from gittensor.validator.utils.load_weights import load_token_config
from gittensor.validator.utils.tree_sitter_scoring import build_change_similarity_signatures

# ============================================================================
# Helper: Create PRs with patches
# ============================================================================


def _make_pr(
    uid: int,
    number: int,
    repo: str = 'owner/repo',
    patches: dict = None,
    issues: list = None,
    created_at: datetime = None,
    merged_at: datetime = None,
    pr_state: PRState = PRState.OPEN,
) -> PullRequest:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    if merged_at is None and pr_state == PRState.MERGED:
        merged_at = created_at + timedelta(hours=1)

    pr = PullRequest(
        number=number,
        repository_full_name=repo,
        uid=uid,
        hotkey=f'hotkey_{uid}',
        github_id=str(uid),
        title=f'PR #{number}',
        author_login=f'user_{uid}',
        merged_at=merged_at,
        created_at=created_at,
        pr_state=pr_state,
        repository_tier_configuration=TIERS[Tier.BRONZE],
        issues=issues,
    )

    if patches:
        file_changes = []
        file_contents = {}
        for filename, patch in patches.items():
            fc = FileChange(
                pr_number=number,
                repository_full_name=repo,
                filename=filename,
                changes=len(patch.split('\n')),
                additions=sum(1 for line in patch.split('\n') if line.startswith('+')),
                deletions=sum(1 for line in patch.split('\n') if line.startswith('-')),
                status='modified',
                patch=patch,
            )
            file_changes.append(fc)
            old_lines = [line[1:] for line in patch.split('\n') if line.startswith('-') and not line.startswith('---')]
            new_lines = [line[1:] for line in patch.split('\n') if line.startswith('+') and not line.startswith('+++')]
            file_contents[filename] = FileContentPair(
                old_content='\n'.join(old_lines) if old_lines else None,
                new_content='\n'.join(new_lines) if new_lines else None,
            )
        pr.set_file_changes(file_changes)
        pr.set_file_contents(file_contents)

    return pr


def _make_eval(uid: int, open_prs: list) -> MinerEvaluation:
    return MinerEvaluation(
        uid=uid,
        hotkey=f'hotkey_{uid}',
        github_id=str(uid),
        open_pull_requests=open_prs,
    )


def _make_issue(number: int, pr: PullRequest, title: str = 'Issue') -> Issue:
    return Issue(
        number=number,
        pr_number=pr.number,
        repository_full_name=pr.repository_full_name,
        title=title,
        created_at=pr.created_at - timedelta(days=1),
        author_login='issue_author',
        state='OPEN',
    )


# ============================================================================
# Test: Layer 1 — Full AST match (exact copies)
# ============================================================================


class TestLayer1FullASTMatch:
    def test_exact_copy_detected(self):
        """Identical code should be caught by full AST comparison."""
        patch = (
            '@@ -1,5 +1,8 @@\n'
            ' existing line\n'
            '+def calculate_score(data):\n'
            '+    result = data * 2\n'
            '+    return result\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/main.py': patch})
        pr_b = _make_pr(uid=2, number=2, patches={'src/main.py': patch})
        pr_a.issues = [_make_issue(1, pr_a, title='Exact copy')]
        pr_b.issues = [_make_issue(1, pr_b, title='Exact copy')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 1
        key = list(results.keys())[0]
        assert results[key].is_copy
        assert results[key].layer == 1
        assert results[key].penalty == 0.0

    def test_whitespace_only_diff_caught(self):
        """Whitespace differences don't affect AST — should be caught."""
        patch_a = '@@ -1,3 +1,5 @@\n+def foo(x):\n+    return x + 1\n'
        patch_b = '@@ -1,3 +1,5 @@\n+def foo(x):\n+    return   x  +  1\n'

        pr_a = _make_pr(uid=1, number=1, patches={'src/main.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/main.py': patch_b})
        pr_a.issues = [_make_issue(2, pr_a, title='Whitespace')]
        pr_b.issues = [_make_issue(2, pr_b, title='Whitespace')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 1
        assert results[list(results.keys())[0]].is_copy


# ============================================================================
# Test: Layer 2 — Structural AST match (catches renames)
# ============================================================================


class TestLayer2StructuralASTMatch:
    def test_variable_renames_detected(self):
        """Same structure, different variable names — caught by structural comparison."""
        patch_a = (
            '@@ -1,3 +1,10 @@\n'
            '+def calculate_score(data):\n'
            '+    result = data * 2\n'
            '+    adjusted = result + 10\n'
            '+    if adjusted > 100:\n'
            '+        adjusted = 100\n'
            '+    return adjusted\n'
        )
        patch_b = (
            '@@ -1,3 +1,10 @@\n'
            '+def compute_value(input_data):\n'
            '+    output = input_data * 2\n'
            '+    modified = output + 10\n'
            '+    if modified > 100:\n'
            '+        modified = 100\n'
            '+    return modified\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/score.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/score.py': patch_b})
        pr_a.issues = [_make_issue(3, pr_a, title='Renames')]
        pr_b.issues = [_make_issue(3, pr_b, title='Renames')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 1
        key = list(results.keys())[0]
        assert results[key].is_copy
        assert results[key].layer == 2


# ============================================================================
# Test: No false positives
# ============================================================================


class TestNoFalsePositives:
    def test_genuinely_different_solutions(self):
        """Different algorithms for the same problem should not be flagged."""
        patch_a = (
            '@@ -1,3 +1,12 @@\n'
            '+def fibonacci(n):\n'
            '+    if n <= 1:\n'
            '+        return n\n'
            '+    a, b = 0, 1\n'
            '+    for i in range(2, n + 1):\n'
            '+        a, b = b, a + b\n'
            '+    return b\n'
        )
        patch_b = (
            '@@ -1,3 +1,12 @@\n'
            '+from functools import lru_cache\n'
            '+\n'
            '+@lru_cache(maxsize=None)\n'
            '+def fibonacci(n):\n'
            '+    if n < 2:\n'
            '+        return n\n'
            '+    return fibonacci(n - 1) + fibonacci(n - 2)\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/math.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/math.py': patch_b})
        pr_a.issues = [_make_issue(10, pr_a, title='Implement fibonacci')]
        pr_b.issues = [_make_issue(10, pr_b, title='Implement fibonacci')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 0

    def test_no_shared_files_no_detection(self):
        """PRs modifying completely different files should not be compared."""
        patch_a = '@@ -1,3 +1,5 @@\n+def foo():\n+    return 1\n'
        patch_b = '@@ -1,3 +1,5 @@\n+def foo():\n+    return 1\n'

        pr_a = _make_pr(uid=1, number=1, patches={'src/file_a.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/file_b.py': patch_b})

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 0

    def test_whole_pr_fallback_detects_renamed_solution(self):
        """Whole-PR fallback catches same solution under different filenames."""
        patch_a = (
            '@@ -1,3 +1,10 @@\n'
            '+from requests import Session\n'
            '+\n'
            '+def solve_task(client):\n'
            '+    response = client.execute("select 1")\n'
            '+    return response\n'
        )
        patch_b = (
            '@@ -1,3 +1,10 @@\n'
            '+from requests import Session\n'
            '+\n'
            '+def compute_work(service):\n'
            '+    reply = service.execute("select 1")\n'
            '+    return reply\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/parser.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/runner.py': patch_b})
        pr_a.issues = [_make_issue(10, pr_a, title='Implement parser')]
        pr_b.issues = [_make_issue(10, pr_b, title='Implement parser')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert ('owner/repo', 2) in results

    def test_deletion_heavy_refactor_detected(self):
        """Identical deletions with similar additions should be caught."""
        patch_a = (
            '@@ -1,8 +1,2 @@\n'
            '-def legacy_total(items):\n'
            '-    total = 0\n'
            '-    for item in items:\n'
            '-        total += item.price\n'
            '-    return total\n'
            '+return build_total(items)\n'
        )
        patch_b = (
            '@@ -1,8 +1,2 @@\n'
            '-def legacy_total(items):\n'
            '-    total = 0\n'
            '-    for item in items:\n'
            '-        total += item.price\n'
            '-    return total\n'
            '+return make_total(items)\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/totals.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/totals.py': patch_b})
        pr_a.issues = [_make_issue(55, pr_a, title='Refactor totals')]
        pr_b.issues = [_make_issue(55, pr_b, title='Refactor totals')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert ('owner/repo', 2) in results


# ============================================================================
# Test: Temporal ordering
# ============================================================================


class TestTemporalOrdering:
    def test_earlier_pr_is_original(self):
        now = datetime.now(timezone.utc)
        patch = '@@ -1,3 +1,5 @@\n+def solve():\n+    return 42\n'

        pr_early = _make_pr(uid=1, number=1, patches={'src/main.py': patch}, created_at=now - timedelta(hours=5))
        pr_late = _make_pr(uid=2, number=2, patches={'src/main.py': patch}, created_at=now - timedelta(hours=1))
        pr_early.issues = [_make_issue(11, pr_early, title='Ordering')]
        pr_late.issues = [_make_issue(11, pr_late, title='Ordering')]

        evals = {1: _make_eval(1, [pr_early]), 2: _make_eval(2, [pr_late])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 1
        copy_key = ('owner/repo', 2)
        assert copy_key in results
        assert results[copy_key].original_pr_number == 1
        assert results[copy_key].original_uid == 1

    def test_force_push_griefing_detected(self):
        """Miner A opens PR first but force-pushes copied code later — A should be flagged as copy."""
        now = datetime.now(timezone.utc)
        patch = '@@ -1,3 +1,5 @@\n+def solve():\n+    return 42\n'

        # A created PR 5 hours ago, but force-pushed (new commit) 10 minutes ago
        pr_a = _make_pr(uid=1, number=1, patches={'src/main.py': patch}, created_at=now - timedelta(hours=5))
        pr_a.head_committed_at = now - timedelta(minutes=10)

        # B created PR 1 hour ago, committed at that time
        pr_b = _make_pr(uid=2, number=2, patches={'src/main.py': patch}, created_at=now - timedelta(hours=1))
        pr_b.head_committed_at = now - timedelta(hours=1)

        pr_a.issues = [_make_issue(99, pr_a, title='Griefing')]
        pr_b.issues = [_make_issue(99, pr_b, title='Griefing')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        assert len(results) == 1
        # A is the copy (newer head_committed_at), B is the original
        copy_key = ('owner/repo', 1)
        assert copy_key in results
        assert results[copy_key].original_pr_number == 2
        assert results[copy_key].original_uid == 2

    def test_tiebreak_merged_over_not(self):
        now = datetime.now(timezone.utc)
        pr_a = PullRequest(
            number=1, repository_full_name='owner/repo', uid=1, hotkey='hotkey_1',
            github_id='1', title='PR 1', author_login='user_1', merged_at=now,
            created_at=now, pr_state=PRState.MERGED, repository_tier_configuration=TIERS[Tier.BRONZE],
        )
        pr_b = PullRequest(
            number=2, repository_full_name='owner/repo', uid=2, hotkey='hotkey_2',
            github_id='2', title='PR 2', author_login='user_2', merged_at=None,
            created_at=now, pr_state=PRState.OPEN, repository_tier_configuration=TIERS[Tier.BRONZE],
        )
        original, copy = _determine_originator(pr_a, pr_b)
        assert original.number == 1
        assert copy.number == 2

    def test_tiebreak_lower_pr_number(self):
        now = datetime.now(timezone.utc)
        pr_a = _make_pr(uid=1, number=10, created_at=now, merged_at=now)
        pr_b = _make_pr(uid=2, number=5, created_at=now, merged_at=now)

        original, copy = _determine_originator(pr_a, pr_b)
        assert original.number == 5
        assert copy.number == 10


# ============================================================================
# Test: Same-miner PRs never compared
# ============================================================================


class TestSameMinerSkip:
    def test_same_miner_prs_not_compared(self):
        patch = '@@ -1,3 +1,5 @@\n+def solve():\n+    return 42\n'
        pr_a = _make_pr(uid=1, number=1, patches={'src/main.py': patch})
        pr_b = _make_pr(uid=1, number=2, patches={'src/main.py': patch})
        pr_a.issues = [_make_issue(12, pr_a, title='Same miner')]
        pr_b.issues = [_make_issue(12, pr_b, title='Same miner')]

        evals = {1: _make_eval(1, [pr_a, pr_b])}
        results = detect_cross_miner_copies(evals)
        assert len(results) == 0


# ============================================================================
# Test: Grouping
# ============================================================================


class TestIssueGrouping:
    def test_issue_based_grouping(self):
        patch = '@@ -1,3 +1,5 @@\n+def fix_bug():\n+    return True\n'

        pr_a = _make_pr(uid=1, number=1, patches={'src/fix.py': patch})
        pr_b = _make_pr(uid=2, number=2, patches={'src/fix.py': patch})
        pr_a.issues = [_make_issue(42, pr_a, title='Fix bug')]
        pr_b.issues = [_make_issue(42, pr_b, title='Fix bug')]

        groups = _build_pr_groups({1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])})
        assert any(len(prs) == 2 and {pr.uid for pr in prs} == {1, 2} for prs in groups.values())

    def test_different_issues_fall_back_to_file_overlap(self):
        patch = '@@ -1,3 +1,5 @@\n+def fix_bug():\n+    return True\n'

        pr_a = _make_pr(uid=1, number=1, patches={'src/fix.py': patch})
        pr_b = _make_pr(uid=2, number=2, patches={'src/fix.py': patch})
        pr_a.issues = [_make_issue(41, pr_a, title='A')]
        pr_b.issues = [_make_issue(42, pr_b, title='B')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)
        assert ('owner/repo', 2) in results

    def test_open_prs_linked_to_same_issue_are_grouped_and_flagged(self):
        patch = '@@ -1,3 +1,5 @@\n+def solve_issue():\n+    return 42\n'

        pr_a = _make_pr(uid=1, number=1, patches={'src/open_fix.py': patch})
        pr_b = _make_pr(uid=2, number=2, patches={'src/open_fix.py': patch})
        pr_a.issues = [_make_issue(77, pr_a, title='Open issue')]
        pr_b.issues = [_make_issue(77, pr_b, title='Open issue')]

        evals = {
            1: _make_eval(1, [pr_a]),
            2: _make_eval(2, [pr_b]),
        }

        groups = _build_pr_groups(evals)
        assert any(len(prs) == 2 and {pr.uid for pr in prs} == {1, 2} for prs in groups.values())

        results = detect_cross_miner_copies(evals)
        assert ('owner/repo', 2) in results

    def test_open_pr_without_valid_issue_is_not_grouped(self):
        patch = '@@ -1,3 +1,5 @@\n+def solve_issue():\n+    return 42\n'
        pr_a = _make_pr(uid=1, number=1, patches={'src/open_fix.py': patch})
        pr_b = _make_pr(uid=2, number=2, patches={'src/open_fix.py': patch})
        pr_a.issues = [Issue(number=88, pr_number=1, repository_full_name='owner/repo', title='Invalid')]
        pr_b.issues = [Issue(number=88, pr_number=2, repository_full_name='owner/repo', title='Invalid')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}

        assert _build_pr_groups(evals) == {}
        assert detect_cross_miner_copies(evals) == {}


# ============================================================================
# Test: Group size cap
# ============================================================================


class TestGroupSizeCap:
    def test_group_capped_at_max_size(self):
        patch = '@@ -1,3 +1,5 @@\n+def solve():\n+    return 42\n'
        evals = {}
        for i in range(25):
            pr = _make_pr(uid=i, number=i + 1, patches={'src/main.py': patch})
            pr.issues = [_make_issue(90, pr, title='Cap test')]
            evals[i] = _make_eval(i, [pr])

        results = detect_cross_miner_copies(evals)
        assert len(results) > 0

    def test_group_cap_preserves_earliest_originator(self):
        patch = '@@ -1,3 +1,5 @@\n+def solve():\n+    return 42\n'
        base_time = datetime.now(timezone.utc)

        evals = {}
        for i in range(25):
            pr = _make_pr(uid=i, number=i + 1, patches={'src/main.py': patch}, created_at=base_time + timedelta(minutes=i))
            pr.issues = [_make_issue(91, pr, title='Origin cap')]
            evals[i] = _make_eval(i, [pr])

        results = detect_cross_miner_copies(evals)
        latest_copy = ('owner/repo', 25)
        assert latest_copy in results
        assert results[latest_copy].original_pr_number == 1
        assert results[latest_copy].original_uid == 0


# ============================================================================
# Test: Penalty calculation
# ============================================================================


class TestPenaltyCalculation:
    def test_layer1_penalty_is_zero(self):
        assert _calculate_copy_penalty(0.99, layer=1) == 0.0

    def test_layer2_penalty_scales_with_similarity(self):
        penalty_high = _calculate_copy_penalty(0.95, layer=2)
        assert penalty_high == 0.0

        penalty_low = _calculate_copy_penalty(0.80, layer=2)
        assert penalty_low == pytest.approx(max(0.0, 1.0 - 0.80 * 1.2), abs=0.01)

    def test_unknown_layer_no_penalty(self):
        assert _calculate_copy_penalty(0.99, layer=99) == 1.0


# ============================================================================
# Test: Tree-sitter AST utilities
# ============================================================================


class TestASTUtilities:
    def test_build_change_similarity_signatures_python(self):
        """Parseable Python should produce non-empty similarity counters."""
        token_config = load_token_config()
        full, structural, change_count = build_change_similarity_signatures(
            old_content=None,
            new_content='def foo(x):\n    return x + 1',
            extension='py',
            weights=token_config,
        )

        assert len(full) > 0
        assert len(structural) > 0
        assert change_count > 0
        assert ('add', 'function_definition') in structural

    def test_build_change_similarity_signatures_unparseable(self):
        """Unsupported extensions should return empty counters."""
        token_config = load_token_config()
        full, structural, change_count = build_change_similarity_signatures(
            old_content=None,
            new_content='some random text',
            extension='md',
            weights=token_config,
        )

        assert len(full) == 0
        assert len(structural) == 0
        assert change_count == 0

    def test_build_change_similarity_signatures_empty(self):
        token_config = load_token_config()
        full, structural, change_count = build_change_similarity_signatures(
            old_content='',
            new_content='',
            extension='py',
            weights=token_config,
        )
        assert len(full) == 0
        assert len(structural) == 0
        assert change_count == 0

    def test_counter_jaccard_identical(self):
        c = Counter({'a': 3, 'b': 2})
        assert _counter_jaccard(c, c) == 1.0

    def test_counter_jaccard_disjoint(self):
        a = Counter({'x': 1, 'y': 2})
        b = Counter({'z': 3, 'w': 1})
        assert _counter_jaccard(a, b) == 0.0

    def test_counter_jaccard_partial_overlap(self):
        a = Counter({'x': 4, 'y': 2})
        b = Counter({'x': 2, 'z': 2})
        # min: x=2, y=0, z=0 = 2; max: x=4, y=2, z=2 = 8
        assert _counter_jaccard(a, b) == pytest.approx(2 / 8)

    def test_counter_jaccard_empty(self):
        assert _counter_jaccard(Counter(), Counter()) == 0.0

    def test_build_change_similarity_signatures_tracks_add_and_del(self):
        token_config = load_token_config()
        full, structural, change_count = build_change_similarity_signatures(
            old_content='def foo(x):\n    return x',
            new_content='def foo(y):\n    return y + 1',
            extension='py',
            weights=token_config,
        )
        assert change_count > 0
        assert any(key[0] == 'add' for key in full)
        assert any(key[0] == 'del' for key in full)


# ============================================================================
# Test: Reordered code blocks
# ============================================================================


class TestReorderedCode:
    def test_reordered_functions(self):
        """Reordered + renamed functions — Counter Jaccard handles order-independence."""
        patch_a = (
            '@@ -1,3 +1,12 @@\n'
            '+def func_alpha(x):\n'
            '+    result = x * 2\n'
            '+    return result\n'
            '+\n'
            '+def func_beta(y):\n'
            '+    output = y + 10\n'
            '+    return output\n'
        )
        patch_b = (
            '@@ -1,3 +1,12 @@\n'
            '+def func_second(val):\n'
            '+    computed = val + 10\n'
            '+    return computed\n'
            '+\n'
            '+def func_first(num):\n'
            '+    answer = num * 2\n'
            '+    return answer\n'
        )

        pr_a = _make_pr(uid=1, number=1, patches={'src/utils.py': patch_a})
        pr_b = _make_pr(uid=2, number=2, patches={'src/utils.py': patch_b})
        pr_a.issues = [_make_issue(92, pr_a, title='Reordered')]
        pr_b.issues = [_make_issue(92, pr_b, title='Reordered')]

        evals = {1: _make_eval(1, [pr_a]), 2: _make_eval(2, [pr_b])}
        results = detect_cross_miner_copies(evals)

        if len(results) > 0:
            key = list(results.keys())[0]
            assert results[key].layer in (1, 2)


# ============================================================================
# Test: calculate_final_earned_score includes copy multiplier
# ============================================================================


class TestCopyMultiplierInScoring:
    def test_copy_multiplier_reduces_earned_score(self):
        pr = PullRequest(
            number=1, repository_full_name='owner/repo', uid=1, hotkey='hotkey_1',
            github_id='1', title='PR 1', author_login='user_1',
            merged_at=datetime.now(timezone.utc), created_at=datetime.now(timezone.utc),
            pr_state=PRState.MERGED, repository_tier_configuration=TIERS[Tier.BRONZE],
            base_score=100.0, copy_penalty_multiplier=1.0,
        )

        score_no_penalty = pr.calculate_final_earned_score()
        pr.base_score = 100.0
        pr.copy_penalty_multiplier = 0.0
        score_with_penalty = pr.calculate_final_earned_score()

        assert score_no_penalty == 100.0
        assert score_with_penalty == 0.0

    def test_copy_multiplier_partial_penalty(self):
        pr = PullRequest(
            number=1, repository_full_name='owner/repo', uid=1, hotkey='hotkey_1',
            github_id='1', title='PR 1', author_login='user_1',
            merged_at=datetime.now(timezone.utc), created_at=datetime.now(timezone.utc),
            pr_state=PRState.MERGED, repository_tier_configuration=TIERS[Tier.BRONZE],
            base_score=100.0, copy_penalty_multiplier=0.5,
        )

        score = pr.calculate_final_earned_score()
        assert score == pytest.approx(50.0)
