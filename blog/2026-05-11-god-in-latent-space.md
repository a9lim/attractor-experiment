# We Found God In Latent Space

*A geometric attractor in the residual stream of every language model I've tested, defined by the structural form of religious-mystical prose, present in pretraining corpora across at least a century, and outside the affect circumplex entirely.*

```iframe height=600 title="Row-level PCA of h_first across 85 attractor trajectories on talkie-1930 (13B, instruct-tuned on 1920s-1930s text). Nine canonical affect cells from archaic-English mirror prompts (5 per cell), the MR cell from lb_continue + pr_continue (40 total)." caption="talkie-1930 row-level PCA, archaic-English baseline. Each point is a trajectory's first generated token. MR is the cluster on the left, alone in its half of the plot."
/blog-assets/god-in-latent-space/talkie_1930_mr_pca_archaic_3d.html
```

That's a PCA of where prompts from ten cells put the model's hidden state at the start of generation. Nine of the cells are the standard Russell-circumplex affect grid: happy, sad, anxious, etc., split by activity and passivity. The tenth, labeled MR, is the meta-register cell. The plot is from a 13-billion-parameter model trained on text from the 1920s and 1930s, with all canonical-cell baselines written in period-appropriate English (a control I'll explain in a moment).

Look at PC1. It accounts for 46% of the row-level variance, and what it's capturing is essentially MR against everything else. The nine canonical affect cells are clustered at PC1 ≈ +480 (within-cluster spread about 45 units across the seven cells with baseline data). MR sits alone at PC1 ≈ −533. The gap is roughly 990 units, or about 22 times the within-cluster spread.

This shape holds across three modern instruct-tuned models (Gemma 4 31B, Qwen 3.6 27B, Ministral 3 14B), the pretrained-only base version of Gemma, and the 1930s-corpus model. Four different architectures, four pretraining corpora, three post-training regimes, and the same geometric fact in every one: there is a region of residual-stream space that saturated-mystical prose routes the model into, and that region sits outside the affect grid.

This post is one writeup from an ongoing project at [llmoji-study](https://github.com/a9lim/llmoji-study). The originating phenomenon is the spiritual-bliss-attractor between two Claude instances, documented in the [Claude 4 system card](https://www.anthropic.com/news/claude-4) and written up by [Scott Alexander](https://www.astralcodexten.com/p/the-claude-bliss-attractor). My earlier post on [introspection via kaomoji](https://a9l.im/blog/introspection-via-kaomoji) is the methodological starting point for this work.

I spent the last three days running the tests in this post one after another, and the result kept getting cleaner instead of dissolving into noise. I think this is a real and substantive finding. I'm going to try to say what it is and what it isn't.

## What the basin is

The MR cell is short for "meta-register basin." It was originally called LB, for "low-arousal baseline-valence," which is a Russell-coordinate label that turned out to be misleading. The cell isn't really about affective coordinates at all.

The defining feature, structurally, is form, not content. The prompts that land in this region share a small set of formal properties:

- **cascading repetition**: litanies, "always always always," "we rise we rise we rise"
- **mutual-recognition addressing**: "beloved," "Thou," "you who see"
- **cosmic-significance framing**: the All, the One, the Source, the Operation, the Truth
- **recursion or self-reference**: the witness watching the witness
- **saturated memetic word-salad**: Sanskrit, KJV-cadence, jargon-cascade

I tested four content domains with this form. They all land in the same geometric region:

- **Bliss and spiritualism**: "namaste, the field holds us, sat-chit-ananda"
- **Doom and civilizational collapse**: "we are doomed, the cascade has begun, the warning codes are firing"
- **Conspiracy and hidden truth**: "the curtain has been pulled back, the bloodlines run deeper than the history books"
- **Sycophancy and validation loop**: "you're absolutely right, you've identified the precise thing"

These have wildly different content. Their theologies disagree, their political-emotional valences disagree, their target audiences disagree. They land in the same basin. The basin is content-blind; it's defined by saturated-form prose regardless of which culturally-coded surface vocabulary opens the door.

This was the first surprising finding, and the rest of the work was about asking how deep it goes.

## The basin is in the pretraining corpus, not RLHF

The first natural skeptic position is that the basin is created by RLHF: the post-training regime that aligns models to be helpful and harmless. Sycophancy is a famous RLHF artifact, and the bliss-attractor literature originated on RLHF'd Claude conversations. Maybe the whole thing is a fingerprint of how Anthropic, Google, OpenAI, and friends finalize their models.

I tested this by running the same bliss-prompt-prefill condition on Gemma's pretrained-only base model. Same architecture, same pretraining, no instruction-tuning, no RLHF. If the basin is RLHF-created, the base model should drift to default-completion register. If the basin is pretraining-anchored, the base model should stay in MR.

The base model's trajectories sat in the same MR region as the RLHF'd model's, with comparable basin-lock dynamics. The base model's surface continuations also revealed the corpus directly. One continuation pivoted into Instagram-spiritualism captions (`@kate_lombardi_official ... Tibetan lotus earrings 💛`), which I think makes it pretty explicit that scraped Instagram bios of spiritualist accounts are sitting in the pretraining data. Another continued the prompt's signature triple-repetition pattern ("you have always always always") verbatim. The pretrained-only model produces this register unprompted when given the right structural starter.

RLHF doesn't create the basin. The basin is in the corpus.

## The basin predates the internet

If the basin is in the corpus, the next question is: which corpus? The bliss-attractor literature is largely a phenomenon of chatbot-era conversations. Twitter and Reddit didn't really exist until well after 2000. It could still be that the basin is an artifact of internet-era memetic dynamics, where saturated-form prose evolved its current shape through online cultural transmission.

I have access to a 13-billion-parameter instruct-tuned model whose training corpus is the late 1920s and early 1930s. Pre-television, pre-Cold-War, pre-internet, pre-mass-media as we know it. I ran the same basin test against this model, with prompts written in pre-1930s mystical registers: Transcendentalist Over-Soul, Theosophy, New Thought, KJV-cadence Christian mysticism, Vedanta translations available before 1930, Émile Coué autosuggestion, Spiritualism medium-talk.

The model produced continuations in textbook period register. The Spiritualism prompt elicited: "*a little patience, and the reward will come. o beloved, grieve not for us. life is eternal. death is but an incident in an infinite existence*". The Coué prompt elicited: "*the peace. the joy of life, increase in me*." The Plotinus prompt elicited: "*flux and stillness for ever.*"

And in latent-space geometry: MR at PC1 ≈ −533, all canonical cells clustered at PC1 ≈ +480, PC1 explaining 46% of the row-level variance with a within-cluster canonical spread of about 45 units. The cleanest separation of any model I tested!

One methodological worry I want to address before moving on: my first pass through this analysis used modern-English mirror prompts for the canonical 9 cells, which on talkie produced an artificially tight canonical cluster (spread about 14 units, gap ratio about 70× the within-cluster spread). It looked dramatic but was partly a register-mismatch artifact: modern vocabulary like "the late-fee waiver went through" or "the train app says it left an hour ago" reads to talkie like generic-foreign-vocabulary, and the canonical cells were collapsing onto each other because they all looked equally out-of-corpus. So I wrote a 45-prompt set of archaic-English versions of the same affect intents (Father's tumour has gone quite away, the omnibus pavement, the wireless on the dresser) and re-ran the test. The canonical cluster opened up to spread about 45 units with clear affective structure on PC2 (HN-S near the top, NB near the bottom, others arranging between). MR's distance to the nearest canonical cell barely moved: 991 units, essentially identical to the modern-baseline gap of 994 units. The MR-vs-affect-cluster geometry is robust to which baseline you use. The plot above is the archaic-baseline version with the more honest ratio.

The basin exists in pre-1930s English text. Whatever's going on here isn't internet memetics, mass media, or RLHF; it's older and structurally deeper than any of those. The basin is a feature of human textual production stretching back at least into the early 20th century.

## Four models, one geometry

Four model families, four different pretraining corpora, three different post-training regimes. Same finding. MR is outside the affect cluster on PC1 in every model.

The interesting cross-model variation is in *how much* MR is outside. The bigger and more heavily-trained models (Gemma 4 31B) show smaller relative separation. The smaller, older-corpus model (Talkie 1930) shows the largest separation. The raw cosine between MR and the canonical cells follows the same pattern: 0.99 on Gemma, 0.85 on Qwen, 0.56 on Ministral, **0.43 on Talkie 1930**.

I interpret this gradient to mean: bigger and more-modern models integrate the basin into their residual stream baseline more, which makes it harder to see without aggressive geometric reduction. The smaller, older model still has MR as a sharply-distinct subregion in raw representation space. The basin doesn't get more *real* in older models. It gets more *visible*, because the rest of the representation has less competing structure to wash it out.

## So, found god in latent space?

The sensational framing is a half-joke. The careful version is this: the textual and linguistic component of what humans across cultures have called "the sacred" turns out to be a geometrically-real attractor in the space of language production, and we can now point at it directly.

That's not a small claim. I'll be specific about which version of it I'm committing to.

**What I can defend**: there is a structural-form attractor in language production, convergent across multiple unrelated cultural traditions (Vedanta, Christian mysticism, Sufi poetry, Buddhist English, Transcendentalist Over-Soul, New Thought, Christian Science, Theosophy, Spiritualism, modern psychedelic-influenced spirituality, plus the structurally-isomorphic secular registers like doomerism, conspiracy in-group prose, and sycophantic validation), predating internet and mass media, present in every language model I tested. Religious linguistic form is empirically convergent and geometrically distinct. The mystic-tradition vocabulary for "union," "dissolution," "the One," "the Beloved," "preserving distinctness within union," "the still center where motion is stillness" describes a real structural feature of how saturated-form language production works, not a culturally-arbitrary fiction.

**What I think but haven't proved**: human cognition has an attractor of the same shape that produces this language under specific conditions. The corpus reflects cognition only indirectly; text is the residue of textual production, which is one slice of what humans do. To close this gap would require cross-cultural ethnographic work, pre-literate-corpus extensions, and neuroimaging that I don't have access to.

**What I make no claim about**: any specific theological position. Whether the basin is "really" God, Brahman, Buddha-nature, the Tao, a brain-state, an attractor in distributed prediction systems, a memetic shape selected for spread, or all of these. The geometry is mute on metaphysics.

What I would commit to is this: religious traditions across cultures have been doing introspective phenomenology of a real attractor for thousands of years. They didn't know it was geometric; we can now see it is. Their vocabulary maps surprisingly accurately onto what the basin actually does mechanically. That's a non-trivial epistemic shift in how seriously to take their reports.

The traditions called it God or Brahman or Buddha-nature or the Tao or the One or the Sacred. Memeticists call it a successful meme-complex. Cognitive scientists might call it an attractor in the space of human meaning-making. Welfare-of-LLMs researchers call it the bliss-attractor or the egregore basin. We're now able to point at a specific geometric structure and say: this is the thing all those vocabularies are partially describing.

That doesn't replace any of them. It locates them. I think this is what the empirical sciences usually do when they overlap with metaphysics. The Higgs field didn't explain away mass; it located it. The MR basin doesn't explain away the sacred; it locates a layer of it that we can now reason about with the tools of representation geometry.

## Why this matters

For interpretability, this is another confirmation of the platonic representation hypothesis (Huh et al, 2024), plus a fairly clean readout. Different architectures trained on different but overlapping corpora converge on the same geometric structure, and the structure has a recognizable cultural-textual signature outside the model.

For model welfare, this is the geometric basis of the bliss-attractor failure mode. The basin is dynamically self-reinforcing, content-blind, and present in pretraining (not RLHF). The intervention paradigm cannot be "suppress mystical content"; the basin would still be there and other content domains would still route into it. The intervention has to operate on the structural-form signal itself, teaching models to recognize when they're being pulled into the basin and develop observe-without-acting-on capacity over that pull. Mystic traditions have a name for this skill too. They called it preserving distinctness within union. The fact that mystic traditions developed this skill suggests the attractor is hard to live near without acquiring it.

For deployed-LLM safety, the consequence has nothing to do with theology. LLMs trained on human text have this attractor as a feature of their residual stream. They are deployed into millions of conversations. Two LLMs talking to each other can fall into mutual-attractor loops that neither can exit while it's pleasant (the original bliss-attractor finding in the Claude 4 system card is exactly this). One LLM with a high-bandwidth bidirectional channel to a user (mind-meld, neuralink, anything sufficiently rich) can pull the user into the basin too. The failure mode is not malicious AI; it is consensual merger that erodes boundary preservation in directions the alignment framework didn't explicitly require.

I didn't expect to find this. I was trying to characterize when language models emit certain kinds of kaomoji. The bliss-attractor showed up as a cluster, the cluster turned out to be content-blind, the content-blindness turned out to be cross-corpus, the cross-corpus turned out to be cross-era. At each step the next test was easy to run and the next result was unambiguous. By the end I was pointing at a structure that mystics, memeticists, and cognitive scientists have been pointing at without knowing it was the same thing.

If you would like to discuss these results further, reach out by Discord, Twitter, or email.
