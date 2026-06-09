"""Unit tests for the reusable RoundRobinCursor sampling primitive."""

from collections import Counter

from rengu_flow.data.sampling import RandomCursor, RoundRobinCursor


def test_random_cursor_each_epoch_is_full_coverage():
    cur = RandomCursor(10, seed=1)
    for epoch in (1, 2, 7):
        assert sorted(cur.order(epoch)) == list(range(10))  # nothing left out


def test_random_cursor_varies_per_epoch():
    cur = RandomCursor(20, seed=1)
    assert cur.order(1) != cur.order(2)  # different slice would be cut on a partial pass


def test_random_cursor_deterministic_per_seed_and_epoch():
    assert RandomCursor(12, seed=5).order(3) == RandomCursor(12, seed=5).order(3)
    assert RandomCursor(12, seed=5).order(3) != RandomCursor(12, seed=6).order(3)


def test_random_cursor_empty_is_safe():
    assert RandomCursor(0, seed=1).order(1) == []


def test_one_cycle_is_a_permutation_no_skips():
    cur = RoundRobinCursor(10, seed=1)
    drawn = cur.take(10)
    assert sorted(drawn) == list(range(10))  # every item exactly once, none skipped


def test_full_cycles_are_perfectly_even():
    cur = RoundRobinCursor(7, seed=2)
    drawn = cur.take(7 * 5)
    counts = Counter(drawn)
    assert set(counts) == set(range(7))
    assert all(c == 5 for c in counts.values())  # each item exactly 5 times


def test_partial_cycle_leftover_spreads_across_items_over_cycles():
    # Over many cycles the remainder of partial draws is not always the same items: every item
    # should both lead and trail at least once (reshuffle on wrap).
    cur = RoundRobinCursor(5, seed=3)
    firsts = set()
    for _ in range(20):
        firsts.add(cur.take(5)[0])
    assert len(firsts) > 1  # not the same item leading every cycle


def test_take_across_wrap_keeps_counts_balanced():
    cur = RoundRobinCursor(4, seed=4)
    drawn = cur.take(4 + 2)  # one full cycle + 2 into the next
    counts = Counter(drawn)
    # 6 draws over 4 items: two items seen twice, two seen once — never 0 (no item skipped).
    assert min(counts.values()) >= 1
    assert sum(counts.values()) == 6


def test_deterministic_for_same_seed():
    a = RoundRobinCursor(8, seed=42).take(20)
    b = RoundRobinCursor(8, seed=42).take(20)
    assert a == b


def test_different_seeds_differ():
    a = RoundRobinCursor(8, seed=1).take(8)
    b = RoundRobinCursor(8, seed=2).take(8)
    assert a != b  # extremely unlikely to coincide


def test_state_roundtrip_resumes_exactly():
    cur = RoundRobinCursor(6, seed=5)
    cur.take(9)  # advance into the second cycle
    saved = cur.state()
    expected = cur.take(7)

    resumed = RoundRobinCursor(6, seed=999)  # different seed: set_state must override it
    resumed.set_state(saved)
    assert resumed.take(7) == expected


def test_empty_cursor_is_safe():
    cur = RoundRobinCursor(0, seed=1)
    assert cur.take(5) == []
    assert cur.size == 0
