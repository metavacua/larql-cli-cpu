"""Cost calibration for the gate's decision parameters.

The gate's error rates are not chosen; they are derived from what errors
cost here, measured in one unit — hours of delay — from this repo's own
history:

- `escapes`:   changes that introduced a defect later fixed, and how long
               the defect lived (the cost of a miss)
- `false_reds`: CI verdicts that flipped on identical code, and how long
               the PR waited for green (the cost of a false alarm)

Every estimate is reported with its sample size and an interval, because
a short history yields wide ones; the decision step must carry that
uncertainty rather than use point values.
"""
