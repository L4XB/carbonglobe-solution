"""Where the ED runs stand inside their site orders.

    python scripts/ed/order_status.py <order file> [<order file> ...]

Prints, per order file, how many of its sites are finished in any fire_cx17b* root and the position of the first unfinished
block; also the overlap of the unfinished parts of the given orders.
"""
import sys
from pathlib import Path

ROOTS = [Path('work/edv3/fire_cx17b'), Path('work/edv3/fire_cx17b_srv')]


def done_sites(root: Path) -> set[int]:
    """Sites with a finished csv in a root."""
    return {int(p.name[4:9]) for p in (root / 'csv').glob('site*.csv')}


def main() -> None:
    done = {r.name: done_sites(r) for r in ROOTS}
    alld = set().union(*done.values())
    print('finished: ' + ', '.join(f'{k} {len(v)}' for k, v in done.items()) + f'; union {len(alld)}; in both {len(set.intersection(*done.values()))}')
    left = {}
    for f in sys.argv[1:]:
        order = [int(x) for x in Path(f).read_text().split()]
        flags = [s in alld for s in order]
        first = flags.index(False) if False in flags else len(order)
        last_done = max((i for i, d in enumerate(flags) if d), default=-1)
        left[f] = [s for s, d in zip(order, flags) if not d]
        print(f'{f}: {len(order)} sites, finished {sum(flags)}, first unfinished at position {first}, last finished at {last_done}, unfinished {len(left[f])}')
        # density of finished sites per block of 100 positions
        print('   finished per 100 positions: ' + ' '.join(str(sum(flags[i:i + 100])) for i in range(0, len(order), 100)))
    fs = list(left)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            print(f'unfinished in both {fs[i]} and {fs[j]}: {len(set(left[fs[i]]) & set(left[fs[j]]))}')


if __name__ == '__main__':
    main()
