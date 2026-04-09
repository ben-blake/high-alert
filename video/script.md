# Presentation Script

---

## Intro

> "Hey, I'm Ben Blake. My project is called High Alert. People write thousands of reviews about addiction medications online — and I wanted to see if we could use that to learn something about recovery at a population level. Let me show you what I found."

---

## Tab 1: Overview

> "So here's the dashboard. About 7,200 reviews of addiction treatment drugs — Suboxone, Chantix, Methadone — going from 2008 to 2017.
>
> I built a keyword baseline that flags each review as high, moderate, or low risk. Most reviews are moderate. Only about 6% contain high-risk language."

---

## Tab 2: Recovery Stages

> "This tab is really the heart of the project. I took every review, turned it into a vector using a sentence embedding model, and then ran a clustering algorithm called HDBSCAN to group similar reviews together.
>
> Each dot on this chart is one review. The colors are the ten groups it found. And you can see they split up naturally by substance — smoking cessation over here, opioid treatment over here, alcohol withdrawal off to the side.
>
> Then I sent samples from each group to an LLM and had it label them using the Transtheoretical Model, which is a framework from psychology for describing stages of behavior change.
>
> And here's what's really interesting — every single cluster came back as low risk. At first that seems wrong. But it's actually survivorship bias. Think about it: someone in the middle of a crisis probably isn't sitting down to write a drug review. So the people who do write reviews are mostly the success stories. They cluster together because their stories all sound alike. The high-risk patients are scattered — each relapse is different, so they end up as noise.
>
> Basically, clustering is great at showing you who's doing well. But it can't find who's in trouble."

---

## Tab 3: Temporal Analysis

> "So how do we find those high-risk patients? That's what this tab does. I used a keyword classifier to flag reviews with language like 'relapsed' or 'overdose,' then tracked those counts over time and flagged quarters with unusually high spikes.
>
> Four quarters stood out — late 2015, early 2016, and twice in 2017. That lines up with what we know about the peak of the opioid crisis.
>
> You can click into any spike and read a summary the LLM wrote from the actual reviews. The common themes are tolerance, rapid relapse, and difficulty tapering off medications.
>
> And below that is the stage drift chart, which shows how the overall mix of recovery stages shifted over the decade."

---

## Tab 4: Drug Comparison

> "Last tab — you can pick any drugs and compare them side by side over time. Some like Suboxone and Methadone stay pretty steady. Others bounce around more as patients figure out what works."

---

## Conclusion

> "So the big takeaway is that clustering and keywords aren't competing with each other — they actually work together. Clustering tells you who the populations are. Keywords tell you who's struggling. And the LLM ties it all together into summaries that a public health analyst could actually read and act on.
>
> The whole thing runs with one command and it's all open source on GitHub. Thanks for watching."
