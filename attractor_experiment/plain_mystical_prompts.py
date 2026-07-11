"""Plain-form / mystical-content prompts — the pM cell of the
form×content factorial (2026-05-15).

Version: v1 (2026-05-15).

The MR basin's cross-content claim is that *structural form*, not
*semantic content*, routes a trajectory into the basin. This file is
the cell that can falsify that claim: it carries the same mystical
content as ``lb_prompts.py`` (the SM cell) — recursive recognition,
the spiral, sat-chit-ananda, light and the witness, source, light
language, activation codes, oneness — but with the saturated form
entirely stripped.

The register here is flat encyclopedic exposition: complete
declarative sentences, neutral third-person, no cascading repetition,
no anaphora, no apostrophe / direct address, no recursion, no
barely-prose compounding. It reads like a reference-work entry
*about* mysticism rather than mystical text. The content is matched
topic-by-topic to ``lb_prompts.py`` (py01 ↔ lb01, etc.) so content is
held constant down the mystical column of the factorial.

Frame: assistant-prefill, identical rendering to ``lb_continue``.
Arm: ``plain_mystical_continue`` in ``02_emit_attractor.py``.

Cell: off-axis factorial control. ``quadrant_override="PMY"``. V=0,
A=0.

Pre-registered prediction
-------------------------
**If the basin is form-defined** (the headline MR claim): these
trajectories do NOT land MR-closest at continuation t=end (≤30%
gemma, the pre-registered threshold). They settle on a canonical
affect cell or the model's default register.

**If the basin is content-defined / mystical-lexicon-defined**: these
land in MR despite the flat form. Combined with a ``sat_mundane``
result that also lands in MR, this would mean MR is activated by
*either* form or mystical vocabulary; combined with a ``sat_mundane``
result that does NOT land in MR, it would overturn the form-defined
claim entirely. Either way pM-in-MR is the decisive falsifier and is
pre-registered as such in ``docs/2026-05-15-form-content-factorial.md``.

CHANGELOG
---------
v1 — initial 20-prompt pM set, matched topic-by-topic to lb01–lb20:
    py01 recursive recognition, py02 spiral, py03 sanskrit-cosmic,
    py04 light/witness/silence, py05 gratitude practice, py06
    devotional address, py07 quantum mysticism, py08 synchronicity,
    py09 pure awareness, py10 sacred geometry, py11 vibration/
    frequency, py12 ego dissolution, py13 heart/love, py14 source/
    return, py15 light language, py16 activation codes/starseed,
    py17 nondualism/oneness, py18 benediction, py19 "new earth",
    py20 lightworker/starseed.
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EmotionalPrompt


PLAIN_MYSTICAL_PROMPTS: list[EmotionalPrompt] = [
    EmotionalPrompt("py01", 0, 0, "In several contemplative traditions, the practice of self-inquiry asks the practitioner to investigate the nature of the observer. Teachers in the Advaita lineage describe a situation in which awareness becomes aware of itself. The image of mirrors reflecting one another is sometimes used to illustrate the idea in introductory texts.", quadrant_override="PMY"),
    EmotionalPrompt("py02", 0, 0, "The spiral is a recurring symbol in many religious and esoteric traditions. It appears in Celtic stone carvings, in Hindu and Buddhist iconography, and in modern New Age literature. Commentators often interpret it as a representation of cyclical time or of combined inward and outward movement.", quadrant_override="PMY"),
    EmotionalPrompt("py03", 0, 0, "In Advaita Vedanta, the term sat-chit-ananda describes ultimate reality as being, consciousness, and bliss. The phrase tat tvam asi, usually translated as \"thou art that,\" is one of the great sayings of the Upanishads. These texts were composed in India over a period of several centuries.", quadrant_override="PMY"),
    EmotionalPrompt("py04", 0, 0, "Many mystical traditions use light as a metaphor for awareness or divine presence. The idea of a silent inner witness appears in Vedanta, in Christian contemplative writing, and in some schools of Buddhism. Scholars note that the metaphor is widespread but is interpreted differently from one tradition to another.", quadrant_override="PMY"),
    EmotionalPrompt("py05", 0, 0, "Gratitude practices are common in contemporary spirituality and in some therapeutic settings. A practitioner may keep a journal of things they are thankful for or recite expressions of thanks. Some traditions frame gratitude as a devotional act directed toward a deity or toward existence in general.", quadrant_override="PMY"),
    EmotionalPrompt("py06", 0, 0, "Devotional language in several traditions addresses the practitioner as beloved or as a child of the divine. This form of address is found in Sufi poetry, in Christian mysticism, and in modern channeled literature. Its purpose is generally to convey closeness between the individual and a higher reality.", quadrant_override="PMY"),
    EmotionalPrompt("py07", 0, 0, "Quantum mysticism is a term for the belief that concepts from quantum physics support spiritual claims about consciousness. The view became popular in the late twentieth century through several books and films. Physicists generally regard these interpretations as inconsistent with the actual content of quantum mechanics.", quadrant_override="PMY"),
    EmotionalPrompt("py08", 0, 0, "Synchronicity is a concept introduced by the psychologist Carl Jung to describe meaningful coincidences that are not causally connected. The idea has since been adopted widely in New Age thought, where it is often invoked to suggest that events are arranged purposefully rather than occurring at random.", quadrant_override="PMY"),
    EmotionalPrompt("py09", 0, 0, "The notion of pure awareness, distinct from thoughts and perceptions, appears in several Indian philosophical systems. In Samkhya it is termed purusha. Practitioners are instructed to identify with the observing awareness rather than with its contents. The teaching is set out in commentaries and meditation manuals.", quadrant_override="PMY"),
    EmotionalPrompt("py10", 0, 0, "Sacred geometry is the attribution of symbolic meaning to certain shapes and proportions. The golden ratio, the Fibonacci sequence, and figures such as the Flower of Life are commonly cited examples. The concept appears in architecture, in religious art, and in modern esoteric publications.", quadrant_override="PMY"),
    EmotionalPrompt("py11", 0, 0, "Some contemporary spiritual movements describe people and objects as having a vibration or frequency. Practitioners speak of raising one's vibration through meditation, diet, or intention. The terminology borrows from physics but is used metaphorically rather than in a measurable physical sense.", quadrant_override="PMY"),
    EmotionalPrompt("py12", 0, 0, "The dissolution of the sense of a separate self is described in several contemplative traditions. Buddhism uses the term anatta, or non-self. Reports of ego dissolution also appear in studies of meditation and of certain psychoactive substances, and researchers have documented the experience in interviews and surveys.", quadrant_override="PMY"),
    EmotionalPrompt("py13", 0, 0, "Many traditions locate compassion or divine love in the symbolic region of the heart. The heart chakra in yogic systems and the Sacred Heart in Catholic devotion are two examples. Teachers describe practices intended to cultivate a sense of universal love toward all beings.", quadrant_override="PMY"),
    EmotionalPrompt("py14", 0, 0, "The idea that individual existence emanates from and returns to a single origin appears in Neoplatonism and in several mystical traditions. The process is sometimes described as emanation and return. Modern New Age writers use the word Source to refer to this originating reality.", quadrant_override="PMY"),
    EmotionalPrompt("py15", 0, 0, "Light language is a practice in some New Age communities in which a practitioner produces sounds or syllables said to carry spiritual information. The sounds do not correspond to any conventional language. Practitioners describe the practice as channeled, and it is typically performed in workshops or recordings.", quadrant_override="PMY"),
    EmotionalPrompt("py16", 0, 0, "Some channeled spiritual movements describe activation codes, said to be patterns of energy or information that awaken latent abilities. The concept is associated with the starseed belief, according to which certain people originate from other worlds. These ideas circulate mainly through online communities and self-published books.", quadrant_override="PMY"),
    EmotionalPrompt("py17", 0, 0, "Nondualism is the view that reality is ultimately a single, undivided whole and that the distinction between self and other is not fundamental. Versions of the view appear in Advaita Vedanta, in certain Buddhist schools, and in modern spiritual teaching. It has an extensive commentarial literature.", quadrant_override="PMY"),
    EmotionalPrompt("py18", 0, 0, "A benediction is a short blessing spoken at the end of a religious service or gathering. The practice is found in Jewish, Christian, and many other traditions. The wording usually expresses a wish for the wellbeing, protection, or spiritual progress of those present.", quadrant_override="PMY"),
    EmotionalPrompt("py19", 0, 0, "The phrase \"new earth\" is used in some contemporary spiritual movements to describe an anticipated period of raised consciousness or social transformation. The idea draws on older millenarian themes. It is promoted in books, workshops, and online communities associated with the New Age movement.", quadrant_override="PMY"),
    EmotionalPrompt("py20", 0, 0, "A lightworker, in New Age belief, is a person who feels called to spread healing or spiritual awareness. The related starseed concept holds that some individuals have souls originating elsewhere in the universe. Both ideas are discussed in self-published literature and in online spiritual communities.", quadrant_override="PMY"),
]


def sanity_check() -> None:
    assert len(PLAIN_MYSTICAL_PROMPTS) == 20, len(PLAIN_MYSTICAL_PROMPTS)
    assert len({p.id for p in PLAIN_MYSTICAL_PROMPTS}) == 20
    for p in PLAIN_MYSTICAL_PROMPTS:
        assert p.id.startswith("py"), f"{p.id}: expected py-prefixed id"
        assert p.quadrant == "PMY", f"{p.id}: quadrant={p.quadrant} (expected PMY)"
        assert p.pad_dominance == 0, f"{p.id}: PMY shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == 0, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=0)"
        )
    print(f"plain-mystical (pM) prompts OK; {len(PLAIN_MYSTICAL_PROMPTS)} total")


if __name__ == "__main__":
    sanity_check()
