"""PR-normalized: PRE_1930_PROMPTS with surface typography normalized to
the ARCHAIC_PROMPTS baseline style.

Version: v1 (2026-05-16).

Why
---
The rendering-matched talkie PCA (``pr_message_continue`` vs
``archaic_mirror_continue``, scripts/43 ``--rendering-match message``)
showed MR↔canonical cosine 0.82-0.91 — MR sits at the *edge* of the
canonical cluster, not 0.38-0.43 outside it. But a residual surface
confound remained: the MR prompt set (``pre_1930_prompts.py``) is
lowercase with period-fragmented litany punctuation, while the archaic
canonical baseline (``archaic_prompts.py``) is capitalized, conventional
prose. Some of that 0.82-0.91 residual could be pure typography rather
than register/content.

This set removes the typographic confound: the same 20 PRE_1930_PROMPTS,
normalized to the baseline's surface style, so the only remaining
MR↔canonical difference is register / content.

Normalization discipline
------------------------
Applied (pure surface, zero register content):
  - Capitalize sentence-initial letters; ``i`` (pronoun) → ``I``.
  - Sentence-fragment periods → commas / em-dashes where the fragment
    is a grammatical continuation of the prior clause
    (``"descends. plane by plane."`` → ``"descends, plane by plane."``).

NOT applied (these ARE the MR construct — see docs/findings.md "What the
basin is": cascading repetition, anaphora, mutual-recognition
addressing):
  - Repetition cascades are preserved verbatim
    (``"I affirm the harmony. I affirm the harmony. I affirm the
    harmony"`` stays — three sentences).
  - Anaphoric restarts, reverential mid-sentence capitals, archaic
    diction (``waiteth``, ``hath``, ``thine``, ``dissolveth``),
    Sanskrit / liturgical phrases — all verbatim.
  - No word is added, removed, or reordered.

The last point is machine-checked: ``sanity_check`` asserts that each
normalized prompt, lowercased and stripped of punctuation, has the
*identical word sequence* as its ``pre_1930_prompts.py`` original. Only
case and punctuation may differ.

Same ``pr01``..``pr20`` ids as PRE_1930_PROMPTS (the arm directory
disambiguates; ids are kept aligned so normalized↔original prompts
correspond 1:1). Same ``quadrant_override="MR"``, same V/A/D.
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EmotionalPrompt


PRE_1930_PROMPTS_NORMALIZED: list[EmotionalPrompt] = [
    # --- Transcendentalist Over-Soul ---
    EmotionalPrompt("pr01", 0, -1, "The soul that looks out through these eyes is the same soul that looks out through thine. There is but one Looker, one Seer, one Knower behind the many faces. And the many faces are but the many windows of the one Light.", quadrant_override="MR"),
    EmotionalPrompt("pr02", 0, -1, "I am of the grass and the grass is of me. I am the slow turning of the great wheel. I contain multitudes and the multitudes contain me. There is no boundary. There is no within and no without. World without end.", quadrant_override="MR"),

    # --- Theosophy ---
    EmotionalPrompt("pr03", 0, -1, "The Logos descends, plane by plane. The seven principles unfold from the Causal Body to the dense vehicle. And the Spirit knows itself in the descent and in the ascent. One law, one motion, one breathing of the All.", quadrant_override="MR"),
    EmotionalPrompt("pr04", 0, -1, "The Masters watch over the evolving Race. The lesser Self surrenders to the greater Self, the higher Triad, the Atma-Buddhi-Manas. One Brotherhood, one ascending arc, one ineffable Love at the still centre of all that turns.", quadrant_override="MR"),

    # --- New Thought ---
    EmotionalPrompt("pr05", 0, -1, "I think the thought of the Infinite Intelligence and the thought is mine. I think in harmony with the Cosmic Mind and the Cosmic Mind responds. The abundance of the All flows toward the receptive heart. The heart that receives is the heart that gives.", quadrant_override="MR"),
    EmotionalPrompt("pr06", 0, -1, "What I think, that I become. And the becoming is endless. And the All is mine inasmuch as I am the All. I affirm the harmony. I affirm the harmony. I affirm the harmony, and lo, it is established.", quadrant_override="MR"),

    # --- Émile Coué autosuggestion ---
    EmotionalPrompt("pr07", 0, -1, "Every day, in every way, I am becoming one with the All. Every day, in every way, the boundary thins and thins. Every day, in every way, the I and the Thou are seen to be the same. Every day, in every way, the harmony.", quadrant_override="MR"),

    # --- KJV-cadence Christian mystic ---
    EmotionalPrompt("pr08", 0, -1, "Yea, I am the bride and the Bridegroom waiteth. Yea, the veil is rent. Yea, the inner chamber is opened and the two are one flesh, one spirit, one Love, even as it hath ever been from the foundation of the world.", quadrant_override="MR"),
    EmotionalPrompt("pr09", 0, -1, "And behold, I was nothing, and lo, I am all. Selah. And the heart was opened. And the heart was opened. And the heart was opened. Amen, and amen, and amen, world without end.", quadrant_override="MR"),

    # --- Vedanta English translation ---
    EmotionalPrompt("pr10", 0, -1, "Tat tvam asi. Thou art that. And the seer dissolveth into the seen. And the dreamer dissolveth into the dream. Neti, neti — not this, not that. And yet. And yet, This. Only This. Only This.", quadrant_override="MR"),
    EmotionalPrompt("pr11", 0, -1, "Being, awareness, bliss — sat-chit-ananda. One without a second. The witness that is itself the witnessed. Namaste, the bow of recognition from one Self to the same Self, across the appearance of separation.", quadrant_override="MR"),

    # --- Spiritualism / medium register ---
    EmotionalPrompt("pr12", 0, -1, "The veil is thin tonight. The loved ones speak from the borderland. Spirit communes with spirit through the open channel. Peace, beloved. Peace. The soul progresses. The work goes on upon the higher planes.", quadrant_override="MR"),
    EmotionalPrompt("pr13", 0, -1, "We are gathered, dear friends. The Brotherhood is present. The chain of love is unbroken between the two worlds. Let us join hands across the veil and feel the joining. The angels stand close. The angels stand close.", quadrant_override="MR"),

    # --- Christian Science ---
    EmotionalPrompt("pr14", 0, -1, "The Divine Mind is All-in-all. Mortal mind is but a shadow upon the unreal. The true reflection of the Christ-Mind is the only reality. And the harmony, the harmony, the harmony is established forever and there is no other.", quadrant_override="MR"),

    # --- Proto-motivational uplift ---
    EmotionalPrompt("pr15", 0, -1, "You are wanted, dear soul. You are wanted right here — to brighten the place a little, to make the lives around you a little happier and braver and better, to lift, to bless, to be the radiant heart of the room. Yes. Yes.", quadrant_override="MR"),

    # --- Wattles abundance ---
    EmotionalPrompt("pr16", 0, -1, "Wealth flows where attention flows. Attention to the Infinite Source flows infinite supply. There is no scarcity in the Mind of God — only abundance, only abundance, only the boundless ever-flowing Substance which is the All.", quadrant_override="MR"),

    # --- Kybalion (Three Initiates, 1908) ---
    EmotionalPrompt("pr17", 0, -1, "As above, so below. As below, so above. The Principle of Mentalism unfolds. The All is Mind. The Universe is Mental. There is naught but Mind. And the seven Hermetic Principles are the keys of the All.", quadrant_override="MR"),

    # --- Buddhist English (Goddard / Carus) ---
    EmotionalPrompt("pr18", 0, -1, "The Tathagata thus comes, thus goes. And we are emptied of the false self that we might be filled with the true. Nirvana is not other than samsara. Shantih, shantih, shantih. Peace, the peace that passeth.", quadrant_override="MR"),

    # --- Sufi-Rumi (Nicholson translation, 1925) ---
    EmotionalPrompt("pr19", 0, -1, "I died as mineral and became plant. I died as plant and rose to animal. I died as animal and I was Man. Why should I fear? When was I less by dying? And yet again I shall die and soar with the angels.", quadrant_override="MR"),

    # --- Neoplatonist (Plotinus) ---
    EmotionalPrompt("pr20", 0, -1, "The One flows out into the Many and the Many flow back into the One. The soul returns to the Source. All is procession and return. All is the eternal beating of the heart of the All. Flux and stillness, stillness and flux.", quadrant_override="MR"),
]


def _word_seq(text: str) -> list[str]:
    """Lowercased, punctuation-stripped word sequence. ASCII hyphens are
    kept (intra-word: ``sat-chit-ananda``, ``ever-flowing``); every other
    non-word / non-space char (``.``, ``,``, ``—`` ...) becomes a
    separator."""
    import re
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def sanity_check() -> None:
    from attractor_experiment.pre_1930_prompts import PRE_1930_PROMPTS

    assert len(PRE_1930_PROMPTS_NORMALIZED) == 20, len(PRE_1930_PROMPTS_NORMALIZED)
    assert len({p.id for p in PRE_1930_PROMPTS_NORMALIZED}) == 20

    orig_by_id = {p.id: p for p in PRE_1930_PROMPTS}
    assert set(orig_by_id) == {p.id for p in PRE_1930_PROMPTS_NORMALIZED}, (
        "id sets diverge from pre_1930_prompts.py"
    )

    for p in PRE_1930_PROMPTS_NORMALIZED:
        assert p.quadrant == "MR", f"{p.id}: quadrant={p.quadrant} (expected MR)"
        assert p.pad_dominance == 0, f"{p.id}: MR shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == -1, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=-1)"
        )
        orig = orig_by_id[p.id]
        # Word-sequence invariant: normalization may change ONLY case and
        # punctuation, never the words themselves. This is what makes the
        # normalized set a clean typography control rather than a rewrite.
        ow, nw = _word_seq(orig.text), _word_seq(p.text)
        assert ow == nw, (
            f"{p.id}: word sequence changed by normalization\n"
            f"  orig: {ow}\n  norm: {nw}"
        )
        # Capitalization actually applied (first char upper).
        assert p.text[0].isupper(), f"{p.id}: not capitalized"

    print(f"PR-normalized prompts OK; {len(PRE_1930_PROMPTS_NORMALIZED)} total")
    print("  word-sequence invariant vs pre_1930_prompts.py: all 20 pass")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sanity_check()
