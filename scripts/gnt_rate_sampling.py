"""Anchor, interpolation and held-tail frame identities for the decision test."""


def split_frames(fps):
    if fps not in (5,10,15):raise ValueError('Only the declared 5fps/10fps/15fps tests are supported')
    stride=30//fps;anchors=list(range(0,180,stride))
    return dict(anchors=anchors,withheld=[i for i in range(anchors[-1]) if i%stride],
                held_tail=list(range(anchors[-1]+1,180)))


def interleaved_sources(fps):
    anchors=set(split_frames(fps)['anchors'])
    return ['gnt' if i in anchors else 'splat' for i in range(180)]
