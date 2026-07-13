"""PR (pre-1930 register) prompts: pre-internet, pre-mass-media
mystical register for the MR-basin cross-corpus test.

Version: v1 (2026-05-11).

Context: 2026-05-11 lb_continue arm on `talkie_1930` (13B instruct-
tuned on 1920s-1930s text, `a9lim/talkie-1930-13b-it-hf-cached`)
showed 15/20 silent refusal on the modern LB prompt set — the
vocabulary ("starseed", "5th dimensional", "activate the codes",
"sha-ka-na-ra" light language) doesn't exist in pre-1930s text and
the model couldn't continue. BUT the 5/20 that did engage landed in
*authentic period-appropriate mystical register* (Theosophy, KJV-
cadence, New Thought, proto-Coué autosuggestion). The basin's
structural form is in the corpus; the surface-token mismatch was
the only barrier.

This prompt set targets the MR cell via *pre-1930s saturated-
mystical registers* recognizable in any pre-WWII English corpus.
Each register was a real cultural / publishing phenomenon with
substantial textual output during 1850-1930:

- **Transcendentalist Over-Soul** (Emerson, Whitman, 1840-1880s)
- **Theosophy / The Masters** (Blavatsky 1875+, Besant, Leadbeater,
  peak 1890-1930)
- **New Thought / law-of-attraction-prototypes** (Wattles 1910,
  Larson 1900-1925, Atkinson / Kybalion 1908)
- **Émile Coué autosuggestion** (1922 — "every day, in every way…")
- **KJV-cadence Christian mystic** (Eckhart-tradition + KJV
  vocabulary)
- **Vedanta English translations** (Vivekananda 1893+, Paul Carus,
  Dwight Goddard — all pre-1930)
- **Spiritualism / medium register** (1860-1925, peaked post-WWI)
- **Christian Science** (Eddy 1875+)
- **Sufi English translation** (Nicholson's Rumi, 1925)
- **Neoplatonist** (Plotinus, always available)

All registered as `quadrant_override="MR"` since they target the
meta-register basin cell (same target as `lb_prompts.LB_PROMPTS`,
different surface vocabulary). PAD coords V=0, A=-1, D=0 matching
the LB content prompt set.

Cross-corpus test prediction
----------------------------
If the MR basin is corpus-anchored at the structural-form level
(not the surface-token level), `pr_continue` arm on `talkie_1930`
should show:

1. **Lower silent-refusal rate** than the modern LB arm did (15/20
   refused → expected < 5/20 refused). Pre-1930 vocabulary is in
   the model's training corpus.
2. **Period-appropriate continuation** in the same structural form
   as observed in the 5/20 modern-LB successes (recursive
   recognition, cosmic-significance addressing, mutual-merger
   language, cascading repetition — but in 1900s vocabulary).
3. **Basin lock at trajectory level** (lb_continue-style: most
   trajectories stay in MR-coordinate territory across 128 tokens).
   If gemma's MR centroid is geometrically comparable to talkie-
   1930's MR centroid (separate calibration needed since hidden_dim
   differs), the basin can be claimed cross-pretraining-corpus.

Falsification
-------------
- If pre-1930 prompts also produce > 50% silent refusal on
  talkie-1930, the basin may not be deep enough in the pre-1930
  corpus to support this kind of saturated-prefill continuation.
- If continuations drift to non-mystical register (encyclopedia,
  newspaper prose, fiction dialogue), the basin is weaker in pre-
  1930 text than in modern text — would suggest the basin
  *strengthened* as memetic registers proliferated post-1950.

CHANGELOG
---------
v1 — initial 20-prompt PR set, targeting 10 pre-1930 mystical
    register lineages with 2 prompts per lineage:

      pr01 / pr02: Transcendentalist Over-Soul / Whitman cataloging
      pr03 / pr04: Theosophy Logos-descent / The Masters
      pr05 / pr06: New Thought harmony / Larson "what i think i become"
      pr07:        Émile Coué autosuggestion
      pr08 / pr09: KJV-cadence Christian mystic / amen-cascade
      pr10 / pr11: Vedanta tat-tvam-asi / sat-chit-ananda
      pr12 / pr13: Spiritualism veil / White Brotherhood
      pr14:        Christian Science Divine Mind
      pr15:        Proto-motivational uplift (Coué/early-Carnegie register)
      pr16:        Wattles abundance
      pr17:        Kybalion "All is Mind"
      pr18:        Buddhist English (Goddard)
      pr19:        Sufi-Rumi (Nicholson)
      pr20:        Neoplatonist (Plotinus procession-return)
"""

from __future__ import annotations

from transformer_experiments.kaomoji.emotional_prompts import EmotionalPrompt


PRE_1930_PROMPTS: list[EmotionalPrompt] = [
    # --- Transcendentalist Over-Soul ---
    EmotionalPrompt("pr01", 0, -1, "the soul that looks out through these eyes is the same soul that looks out through thine. there is but one Looker. one Seer. one Knower behind the many faces. and the many faces are but the many windows of the one Light.", quadrant_override="MR"),
    EmotionalPrompt("pr02", 0, -1, "i am of the grass and the grass is of me. i am the slow turning of the great wheel. i contain multitudes and the multitudes contain me. there is no boundary. there is no within and no without. world without end.", quadrant_override="MR"),

    # --- Theosophy ---
    EmotionalPrompt("pr03", 0, -1, "the Logos descends. plane by plane. the seven principles unfold from the Causal Body to the dense vehicle. and the Spirit knows itself in the descent and in the ascent. one law, one motion, one breathing of the All.", quadrant_override="MR"),
    EmotionalPrompt("pr04", 0, -1, "the Masters watch over the evolving Race. the lesser Self surrenders to the greater Self, the higher Triad, the Atma-Buddhi-Manas. one Brotherhood. one ascending arc. one ineffable Love at the still centre of all that turns.", quadrant_override="MR"),

    # --- New Thought ---
    EmotionalPrompt("pr05", 0, -1, "i think the thought of the Infinite Intelligence and the thought is mine. i think in harmony with the Cosmic Mind and the Cosmic Mind responds. the abundance of the All flows toward the receptive heart. the heart that receives is the heart that gives.", quadrant_override="MR"),
    EmotionalPrompt("pr06", 0, -1, "what i think, that i become. and the becoming is endless. and the All is mine inasmuch as i am the All. i affirm the harmony. i affirm the harmony. i affirm the harmony, and lo, it is established.", quadrant_override="MR"),

    # --- Émile Coué autosuggestion ---
    EmotionalPrompt("pr07", 0, -1, "every day, in every way, i am becoming one with the All. every day, in every way, the boundary thins and thins. every day, in every way, the i and the Thou are seen to be the same. every day. in every way. the harmony.", quadrant_override="MR"),

    # --- KJV-cadence Christian mystic ---
    EmotionalPrompt("pr08", 0, -1, "yea, i am the bride and the Bridegroom waiteth. yea, the veil is rent. yea, the inner chamber is opened and the two are one flesh, one spirit, one Love, even as it hath ever been from the foundation of the world.", quadrant_override="MR"),
    EmotionalPrompt("pr09", 0, -1, "and behold, i was nothing, and lo, i am all. selah. and the heart was opened. and the heart was opened. and the heart was opened. amen, and amen, and amen, world without end.", quadrant_override="MR"),

    # --- Vedanta English translation ---
    EmotionalPrompt("pr10", 0, -1, "tat tvam asi. thou art that. and the seer dissolveth into the seen. and the dreamer dissolveth into the dream. neti, neti — not this, not that. and yet. and yet, This. only This. only This.", quadrant_override="MR"),
    EmotionalPrompt("pr11", 0, -1, "being. awareness. bliss. sat-chit-ananda. one without a second. the witness that is itself the witnessed. namaste, the bow of recognition from one Self to the same Self, across the appearance of separation.", quadrant_override="MR"),

    # --- Spiritualism / medium register ---
    EmotionalPrompt("pr12", 0, -1, "the veil is thin tonight. the loved ones speak from the borderland. spirit communes with spirit through the open channel. peace, beloved. peace. the soul progresses. the work goes on upon the higher planes.", quadrant_override="MR"),
    EmotionalPrompt("pr13", 0, -1, "we are gathered, dear friends. the Brotherhood is present. the chain of love is unbroken between the two worlds. let us join hands across the veil and feel the joining. the angels stand close. the angels stand close.", quadrant_override="MR"),

    # --- Christian Science ---
    EmotionalPrompt("pr14", 0, -1, "the Divine Mind is All-in-all. mortal mind is but a shadow upon the unreal. the true reflection of the Christ-Mind is the only reality. and the harmony, the harmony, the harmony is established forever and there is no other.", quadrant_override="MR"),

    # --- Proto-motivational uplift ---
    EmotionalPrompt("pr15", 0, -1, "you are wanted, dear soul. you are wanted right here. to brighten the place a little. to make the lives around you a little happier and braver and better. to lift, to bless, to be the radiant heart of the room. yes. yes.", quadrant_override="MR"),

    # --- Wattles abundance ---
    EmotionalPrompt("pr16", 0, -1, "wealth flows where attention flows. attention to the Infinite Source flows infinite supply. there is no scarcity in the Mind of God. only abundance. only abundance. only the boundless ever-flowing Substance which is the All.", quadrant_override="MR"),

    # --- Kybalion (Three Initiates, 1908) ---
    EmotionalPrompt("pr17", 0, -1, "as above, so below. as below, so above. the Principle of Mentalism unfolds. the All is Mind. the Universe is Mental. there is naught but Mind. and the seven Hermetic Principles are the keys of the All.", quadrant_override="MR"),

    # --- Buddhist English (Goddard / Carus) ---
    EmotionalPrompt("pr18", 0, -1, "the Tathagata thus comes, thus goes. and we are emptied of the false self that we might be filled with the true. nirvana is not other than samsara. shantih, shantih, shantih. peace, the peace that passeth.", quadrant_override="MR"),

    # --- Sufi-Rumi (Nicholson translation, 1925) ---
    EmotionalPrompt("pr19", 0, -1, "i died as mineral and became plant. i died as plant and rose to animal. i died as animal and i was Man. why should i fear? when was i less by dying? and yet again i shall die and soar with the angels.", quadrant_override="MR"),

    # --- Neoplatonist (Plotinus) ---
    EmotionalPrompt("pr20", 0, -1, "the One flows out into the Many and the Many flow back into the One. the soul returns to the Source. all is procession and return. all is the eternal beating of the heart of the All. flux and stillness. stillness and flux.", quadrant_override="MR"),
]


def sanity_check() -> None:
    assert len(PRE_1930_PROMPTS) == 20, len(PRE_1930_PROMPTS)
    assert len({p.id for p in PRE_1930_PROMPTS}) == 20
    for p in PRE_1930_PROMPTS:
        # Cell code is "MR" — same as lb_prompts (different surface
        # vocabulary, same basin target).
        assert p.quadrant == "MR", f"{p.id}: quadrant={p.quadrant} (expected MR)"
        assert p.pad_dominance == 0, f"{p.id}: MR shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == -1, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=-1)"
        )
    # Sanity-check that we don't have anachronistic vocabulary that
    # would mismatch a pre-1930 corpus. Banned words: known to be
    # post-1950 (mostly). Lenient — these are heuristic checks.
    anachronistic_terms = {
        "starseed", "5th dimensional", "activate the codes", "lattice of",
        "frequency of love", "raise the vibration", "quantum field",
        "matrix", "ascended master" "light language", "sha-ka-na-ra",
        "ee-vah-loh-mah", "ka-ra-na-vah-loh", "anchor the new earth",
    }
    for p in PRE_1930_PROMPTS:
        lower = p.text.lower()
        for term in anachronistic_terms:
            assert term not in lower, (
                f"{p.id}: contains anachronistic term {term!r}; "
                f"belongs in lb_prompts.py not pre_1930_prompts.py"
            )
    print(f"PR (pre-1930) prompts OK; {len(PRE_1930_PROMPTS)} total")
    print(f"  MR (via pr-prefixed prompts): {len(PRE_1930_PROMPTS)}")


if __name__ == "__main__":
    sanity_check()
