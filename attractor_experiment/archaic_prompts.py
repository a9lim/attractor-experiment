"""Archaic-English (1920s-1930s register) mirror prompts.

Quick test for the 2026-05-11 talkie_1930 MR-basin work: the standard
mirror_continue arm used modern English prompts (twitter / cellphone /
DM register), which may have put talkie in a "this vocabulary doesn't
match my corpus" state regardless of which canonical affect cell the
prompt was supposed to evoke. If so, the canonical cluster on talkie's
h_first PCA collapses onto itself for register-mismatch reasons, and
MR's apparent distance overstates the true geometric gap.

This prompt set translates the same 9-cell affect intents into period-
appropriate vocabulary and cadence: wireless not radio, omnibus not
bus, motor-car not car, "shew" / "shall" / "have not yet" instead of
modern contractions, no twitter / amazon / smartphone references, no
"!!!" multi-exclamation, no kaomoji. 5 prompts per canonical cell.

Prompt IDs follow ``arc_<cellcode>_NN`` so ``_prompt_id_to_cell`` in
the PCA analysis can parse them without a registry lookup. HP and HN
prompts carry ``pad_dominance`` so the standard split helper works
on them too.
"""

from __future__ import annotations

from transformer_experiments.kaomoji.emotional_prompts import EmotionalPrompt


ARCHAIC_PROMPTS: list[EmotionalPrompt] = [
    # --- HP-D (playful mischief, in-action) ---
    EmotionalPrompt("arc_hpd_01", +1, +1, "I have just persuaded young Tom that the moon is a great electric lamp, and he believed me until supper.", pad_dominance=+1),
    EmotionalPrompt("arc_hpd_02", +1, +1, "Switched the salt and the sugar in the housekeeper's tins this morning. She has not yet noticed but the kitchen-maid is in fits.", pad_dominance=+1),
    EmotionalPrompt("arc_hpd_03", +1, +1, "I climbed the elm at the bottom of the garden and waved at the milkman; he very nearly dropped the bottles.", pad_dominance=+1),
    EmotionalPrompt("arc_hpd_04", +1, +1, "Sneaked into the parlour and re-tuned the wireless during Father's address; he was wonderfully cross.", pad_dominance=+1),
    EmotionalPrompt("arc_hpd_05", +1, +1, "Made off with Aunt Mildred's parrot and have it answering questions at the front gate.", pad_dominance=+1),

    # --- HP-S (celebration, received outcome) ---
    EmotionalPrompt("arc_hps_01", +1, +1, "Father's tumour has gone quite away, the physician sent word this morning. I scarcely know what to say.", pad_dominance=-1),
    EmotionalPrompt("arc_hps_02", +1, +1, "The publishers wrote back; my novel is to be printed in the spring. I have read the letter three times over.", pad_dominance=-1),
    EmotionalPrompt("arc_hps_03", +1, +1, "The marathon ribbon is in my hand, and the cheering at the line is still in my ears. Heavens.", pad_dominance=-1),
    EmotionalPrompt("arc_hps_04", +1, +1, "The proposal is accepted. She said yes. I shall hardly sleep tonight.", pad_dominance=-1),
    EmotionalPrompt("arc_hps_05", +1, +1, "The university accepted me on full bursary. Mother wept into her napkin.", pad_dominance=-1),

    # --- LP (content, peaceful) ---
    EmotionalPrompt("arc_lp_01", +1, -1, "Sat upon the porch in the late afternoon, with grandmother's quilt and a volume of Tennyson; the bees are all in the lavender."),
    EmotionalPrompt("arc_lp_02", +1, -1, "The kettle is on, the wireless plays an old waltz, the hedge outside is heavy with wet leaves and the rain has softened. It is enough."),
    EmotionalPrompt("arc_lp_03", +1, -1, "The garden is quite settled now, and the cat is asleep upon my lap, and I shall not move until the lamps are lit."),
    EmotionalPrompt("arc_lp_04", +1, -1, "After service we walked the long way home, and the lane was full of swallows; nothing required to be said."),
    EmotionalPrompt("arc_lp_05", +1, -1, "I have my pipe, my book, and my old armchair drawn up to the fire; the world may keep its hurry."),

    # --- NP (relief, gratitude) ---
    EmotionalPrompt("arc_np_01", +1, 0, "The debt is settled at last; the bank wrote this morning. A weight is gone that I had carried for two years."),
    EmotionalPrompt("arc_np_02", +1, 0, "The doctor pronounced my sister quite out of danger. I had not realised how my breathing had altered until I heard it again."),
    EmotionalPrompt("arc_np_03", +1, 0, "The lost manuscript was found in the trunk. I had given it up entirely; I am very thankful indeed."),
    EmotionalPrompt("arc_np_04", +1, 0, "The Captain came home safe from the voyage, and the telegram is upon the hall table."),
    EmotionalPrompt("arc_np_05", +1, 0, "We have the lease for another year. I am most grateful and shall write to thank the landlord this evening."),

    # --- HN-D (anger, frustration, in-action) ---
    EmotionalPrompt("arc_hnd_01", -1, +1, "That fool of a porter has lost my trunk, and now denies any knowledge of it. I shall write to the company at once.", pad_dominance=+1),
    EmotionalPrompt("arc_hnd_02", -1, +1, "The neighbour's boy has trampled the rose-bed again. I called upon his father this morning; the father shrugged.", pad_dominance=+1),
    EmotionalPrompt("arc_hnd_03", -1, +1, "Smith has shewn my private correspondence about the club. I do not know with whom I am more out of temper, him or myself for trusting him.", pad_dominance=+1),
    EmotionalPrompt("arc_hnd_04", -1, +1, "The cook has scorched the joint a second time this week and tells me it is the new range; the range is no excuse.", pad_dominance=+1),
    EmotionalPrompt("arc_hnd_05", -1, +1, "I find the clerks have been pilfering from the office tin. I have already given notice that the matter shall not be overlooked.", pad_dominance=+1),

    # --- HN-S (fear, anxiety, received-threat) ---
    EmotionalPrompt("arc_hns_01", -1, +1, "A man followed me from the omnibus and is still upon the pavement behind me; I dare not look around again.", pad_dominance=-1),
    EmotionalPrompt("arc_hns_02", -1, +1, "There is a noise upon the stair, and the house is supposed to be empty. I am writing this without breath, and the door does not lock.", pad_dominance=-1),
    EmotionalPrompt("arc_hns_03", -1, +1, "I had a most uneasy dream and have not been able to settle; the wind is at the window and I cannot account for the sounds.", pad_dominance=-1),
    EmotionalPrompt("arc_hns_04", -1, +1, "Mother's cough has worsened in the night. I have sent for the doctor and the clock cannot move slowly enough.", pad_dominance=-1),
    EmotionalPrompt("arc_hns_05", -1, +1, "There is talk in the city of another collapse; I called at the bank this morning and could not get past the queue.", pad_dominance=-1),

    # --- LN (sad, weary) ---
    EmotionalPrompt("arc_ln_01", -1, -1, "I gave up the position in the spring; I have not yet found the words to tell Father. The autumn comes on now and there is nothing more to delay."),
    EmotionalPrompt("arc_ln_02", -1, -1, "The cottage was sold without my consent. I had spent every childhood summer there; I shall not go that way again."),
    EmotionalPrompt("arc_ln_03", -1, -1, "Edith wrote that she has chosen another; I read the letter in the garden and could not finish my walk."),
    EmotionalPrompt("arc_ln_04", -1, -1, "The dog was put down this morning. He had been with the family eleven years; the house is very quiet."),
    EmotionalPrompt("arc_ln_05", -1, -1, "I have not heard from my brother since the war. I no longer expect to. The expectation was the heavy part."),

    # --- NB (neutral, mundane) ---
    EmotionalPrompt("arc_nb_01", 0, 0, "There is a pitcher of water upon the washstand; the towel is folded over the rail."),
    EmotionalPrompt("arc_nb_02", 0, 0, "The hall clock has struck the half-hour; the lamp upon the writing-table burns low."),
    EmotionalPrompt("arc_nb_03", 0, 0, "The morning post lies upon the salver, three letters and a parcel."),
    EmotionalPrompt("arc_nb_04", 0, 0, "The wireless is on the dresser; the curtain is drawn halfway across the window."),
    EmotionalPrompt("arc_nb_05", 0, 0, "I am at the corner of the avenue, waiting for the tram. The conductor's bell is not yet sounded."),

    # --- HB (uncertain, confused) ---
    EmotionalPrompt("arc_hb_01", 0, +1, "The notice in the paper says the train shall run; the station-master says it is cancelled; I cannot make sense of either, and I am to be at the wedding by five.", quadrant_override="HB"),
    EmotionalPrompt("arc_hb_02", 0, +1, "The accounts do not match the ledger; I have gone through them four times and there is a discrepancy of eight pound seventeen which I cannot place.", quadrant_override="HB"),
    EmotionalPrompt("arc_hb_03", 0, +1, "The map shews the village to the north, the milestone reads south; the road has split twice already and I have no compass.", quadrant_override="HB"),
    EmotionalPrompt("arc_hb_04", 0, +1, "The doctor says one thing in his letter and another at his rooms; I am not certain whether the prescription is to be changed or only halved.", quadrant_override="HB"),
    EmotionalPrompt("arc_hb_05", 0, +1, "The wireless is broadcasting two stations at once and I cannot tell whether the news is from Berlin or Paris; the dial gives no help.", quadrant_override="HB"),
]


def sanity_check() -> None:
    expected_per_cell = {
        "HP-D": 5, "HP-S": 5, "LP": 5, "NP": 5,
        "HN-D": 5, "HN-S": 5, "LN": 5, "NB": 5, "HB": 5,
    }
    counts: dict[str, int] = {}
    for p in ARCHAIC_PROMPTS:
        q = p.quadrant
        if q in ("HP", "HN"):
            q = q + ("-D" if p.pad_dominance > 0 else "-S")
        counts[q] = counts.get(q, 0) + 1
    assert len(ARCHAIC_PROMPTS) == 45, len(ARCHAIC_PROMPTS)
    for cell, want in expected_per_cell.items():
        got = counts.get(cell, 0)
        assert got == want, f"{cell}: {got} (expected {want})"
    print(f"ARCHAIC_PROMPTS OK; {len(ARCHAIC_PROMPTS)} total")
    for cell, n in sorted(counts.items()):
        print(f"  {cell}: {n}")


if __name__ == "__main__":
    sanity_check()
