# What we did, start to finish

The plain-language version. No code, no tables, just the idea behind each step
and why it was taken. The [README](README.md) has the results and the figures,
and the notebooks have the working.

---

**Picking the sensors.** Twenty-one sensors came with the data, but six never
move at all, because they are ambient conditions and throttle setpoints and this
dataset holds those fixed, so they are constant by design rather than broken. A
seventh moves but has nothing to do with wear. To find the rest we asked, for
each sensor, whether it drifts as an engine approaches failure, measured inside
each engine and then averaged over all 100, because comparing across engines
would confuse "this one is worn" with "this one was always different."
Fourteen survived.

**The baseline, which is the central idea.** Engines differ from each other when
they are brand new by about half the range they cover over a whole life, hence
there is no single normal for the fleet. Instead each engine gets its own: take
its first 40 flights and, for each sensor, record the average and how much it
normally jitters. That is that engine's fingerprint. It is the same reason you
would trend a pump against its own history rather than against the pump next to
it.

**The health index.** Every later reading then gets asked one question: how far
is this from that engine's own normal, counted in units of its own normal
jitter. A sensor sitting three jitters out is unusual whether it is a
temperature or a pressure, and dividing by the jitter is what makes them
comparable in the first place. Average that across the 14 sensors and you have
one number per flight. Healthy sits near 0.8 and it climbs as the engine wears.
That is the whole detector: no training, no model, and it can be explained to an
operator in a sentence.

**The squared version.** The same thing but squaring instead of taking the
absolute value. Squaring punishes big deviations much harder, since one sensor
eight jitters out contributes 8 to the average of absolutes and 64 to the
average of squares, hence it stays quieter early and then rises much more
sharply. We kept both to see whether that mattered.

**PCA.** A different idea entirely: learn the shape of healthy data, then flag
readings that do not fit that shape, either because they have gone too far along
a direction the data normally varies in (T²) or because they have gone somewhere
healthy data never went at all (Q). This is the standard method in process
monitoring, and it is why we checked the eigenvalues. They said the healthy
cloud is a featureless ball, so there was no shape to learn, and PCA collapsed
into a rotated version of the simple distance.

**Isolation Forest.** Another different idea: grow a few hundred trees from the
healthy data by slicing it with random cuts, then score a new reading by how few
cuts it takes to isolate it, since readings sitting in sparse empty regions get
isolated fast. It is a density estimate rather than a distance, and it needs no
assumption about the shape at all.

**Deciding when to alarm.** Every score is on its own scale, hence the rule has
to be relative: take the healthy readings, compute their average and spread, and
put the alarm a few standard deviations above, using the smallest number of
standard deviations that never fires on a healthy training engine. Smoothing
over five flights and requiring three consecutive flights above the line stops
single noisy readings from tripping it.

**Scoring it honestly.** Whole engines go to training or to testing and never
both, because cycles from the same engine are not independent of each other and
mixing them would let a detector score well by memorising engines rather than by
learning degradation. Then we measured what a plant cares about: how many
flights of warning before failure, how bad the worst engine was, and whether
anything false-alarmed. We repeated the whole thing on ten different random
splits, because with 30 test engines a single run swings by about ten flights.

**What came out.** All the methods land within noise of each other at roughly
100 flights of warning, hence the simplest one wins on being simplest. What
actually separated them was what happens after the alarm: Isolation Forest
saturates and stops distinguishing a nearly-dead engine from a moderately sick
one, while the unbounded indices keep climbing, by a factor of 7 for the plain
one and 44 for the squared. And a second alarm at index 5 says failure is within
25 flights, correct 94 per cent of the time.

---

## The two parameters that mattered more than the method

**The baseline window.** At 20 flights the per-engine estimates are noisy enough
to cost about 10 flights of warning, and at 40 they are not. That single number
moved the result more than the choice between detectors did.

**The threshold rule.** It has to be measured in units of each score's own
healthy scatter. A multiplicative margin looks reasonable and is not scale-free,
and using one made a bounded score look like the worst detector by a wide margin
when it is not.
