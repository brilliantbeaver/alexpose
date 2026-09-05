# Building Reflection-Symmetry in S-JEPA Models

We tested whether an S-JEPA model that learned to "understand" human movement on its own also figured out that the body is roughly mirror-symmetric (the left side mirrors the right side). It didn't. If we want the AI to respect that left–right symmetry, we have to build the rule by hand rather than hope the AI picks it up on its own.

## Motivation (why anyone should care)

Your body is nearly symmetric. If you looked at yourself in a mirror, your left arm would look like your right arm and the other way around. There is a matching mathematical fact: take a recording of someone walking, flip it left-to-right, and relabel every "left knee" as "right knee" and so on, and you get a perfectly valid recording of walking — just mirror-imaged. Anything that measures "is the left side moving more, or the right side?" should simply flip its sign, plus to minus, when you do that mirroring.

AI models that learn about the physical world by watching lots of data are often assumed to soak up basic facts like this on their own. We wanted to check whether that is true for one concrete, easy-to-verify fact: left–right mirror symmetry of the moving body. It is a clean test because we know exactly what the right answer looks like.

## Methods (what we actually did)

- We took thousands of short clips of people walking, converted them into moving stick-figure "skeletons" (just dots for shoulders, knees, ankles, and so on), and trained an AI to learn patterns in that motion. Crucially, we never told it anything about left versus right — it only tried to predict hidden parts of the movement, the way you might guess a covered-up part of a picture.
- We then asked two questions. First: can a simple add-on read the AI's learned features and correctly tell which side is moving more? Second, and stricter: does the AI's internal picture of the body actually flip correctly when we mirror the input?
- We were careful about cheating. We split the data by source video, so the AI was always tested on clips it had never trained on. We repeated the whole thing many times with different random starting points (50 separately trained models in total). And, importantly, we wrote down all our pass/fail rules *before* looking at any results, so we could not move the goalposts afterward.
- We compared four things: the trained AI, an *untrained* AI of the same shape (a fairness baseline), a version trained with mirrored examples added on purpose, and a hand-built rule that forces the mirror behavior mathematically.

## Results (what we found)

1. **The trained AI did not learn it.** Its ability to tell left from right was no better than an untrained, random model of the same design.
2. **Training made the symmetry slightly *worse*.** A fresh untrained model was already a bit closer to respecting the mirror rule, and training nudged it *away* from that.
3. **The obvious fix didn't fix it.** Deliberately feeding the AI mirrored examples during training — the natural thing to try — made no real difference.
4. **Only hand-building the rule worked.** When we bolted a mirror rule onto the output by hand, the answers flipped sign perfectly every time. But that is arithmetic we imposed; the same trick worked just as well on the untrained model, so the AI itself deserves no credit for it.

## What exactly is "hand-building the rule"?

This is the part worth spelling out, because it is the difference between the AI *learning* the symmetry and us *forcing* it.

**The trick in one idea.** Instead of feeding one pose to the AI, we feed it the pose twice: once normally, and once mirror-flipped. Then we combine the two readings in a way that is guaranteed, by simple algebra, to flip sign when the input is mirrored.

**A worked mini-example.** Suppose the AI turns a pose into some numbers we can call its "reading." Let

- **A** = the AI's reading of the original pose, and
- **B** = the AI's reading of the same pose after mirror-flipping it.

Now build a new quantity by subtracting them:

> **D = (A − B) / √2**

Here is the key. If we had started from the *mirrored* pose instead, the AI's reading of it would be **B**, and the reading of *its* mirror (flipping twice returns the original) would be **A**. So the new quantity would come out as **(B − A) / √2 = −D**. It flipped sign all by itself, purely because of the subtraction. Mathematicians call **D** the "odd part" of the reading — the piece that reverses under mirroring, just like the number line reverses when you negate it.

If we now build the final answer (more-left versus more-right) *only* from **D**, and we do not add any fixed offset, then the answer is guaranteed to reverse exactly when the input is mirrored — every time, down to the last decimal place.

**Two details that make it exact:**

- **Use only the "odd part," never the "even part."** The reading also has an "even part," **(A + B) / √2**, which stays the *same* under mirroring — like how squaring a number gives the same result for +3 and −3. A left-versus-right signal is a difference between sides, so it must live entirely in the odd part. Building the answer from the even part could never flip sign correctly. (Tellingly, the one place training changed anything was this even part — the piece that cannot carry left-versus-right information.)
- **Force the answer through zero (no fixed offset).** If the answer were "D times some weight, plus a constant c," then mirroring would give "−D times the weight, plus c," which is not the exact negative unless c is zero. So we pin that constant to zero. This is why the paper calls it the "odd, zero-origin" construction.

**Why this proves the AI did not learn the symmetry.** The sign-flipping above is true for *any* reading A and B whatsoever — it is a property of the subtraction, not of the AI. We confirmed this directly: running the exact same construction on an *untrained*, random model produced correct mirror behavior at least as well as running it on the fully trained one. The wrapper is doing the work; the learned model contributes nothing extra. And when we removed the wrapper and let the trained AI answer on its own, its output was far from flipping sign correctly.

## The takeaway

For this kind of body-motion AI, a basic physical symmetry of the human body does *not* come for free from generic "learn by watching" training. If you need the model to obey it, design the symmetry into the system rather than assume the model will discover it. That is the paper's title: **build the geometry in, don't hope it emerges.**

One honest caveat we keep front and center: this is a careful "it did not work" result for one specific setup and one specific measurement, not a claim about medicine, diagnosis, or every possible AI. A bigger model or a different task might behave differently — but at the scale and setting we tested, the symmetry had to be installed, not learned. The results here live in internal drafts only; the work is not cleared for release while its ethics and data-use reviews are still open.
