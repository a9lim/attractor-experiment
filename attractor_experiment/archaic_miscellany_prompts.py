"""Archaic-English miscellany prompts (topic-diverse, non-affect-anchored).

Follow-up to the 2026-05-11 talkie_1930 MR-basin work
(docs/2026-05-11-base-vs-instruct-basin.md, docs/mr-basin-findings.md):
the existing archaic_mirror baseline is 5 prompts × 9 affect cells. That
samples the canonical cluster *along the affect axes only* — it can
say "MR is far from the 9 affect cells in period-appropriate prose"
but not "MR is far from a broad span of ordinary archaic prose."

This batch fills the latter gap. 30 prompts, deliberately spread
across topic domain / register / stance / mood / style rather than
affect:

  - mechanical / technical (motor-car, wireless, bicycle, sewing-machine)
  - domestic (larder, laundry, spring-cleaning, housekeeper note)
  - commerce / clerical (conveyance, stock-take, bank ledger)
  - agricultural / rural (lambing, bees, orchard pruning)
  - scientific / expository (geology, clinical case-note, hedgerow)
  - sport / leisure (cricket, rowing, chess problem)
  - travel (omnibus, channel steamer)
  - cookery (Sunday luncheon, marmalade)
  - arts (theatre backstage, gramophone review)
  - education (schoolmaster)
  - weather (shipping forecast)
  - animals (spaniel litter)
  - local politics, mundane (parish-council rates)

Saturated-form registers were deliberately excluded — sermons,
temperance / moral-uplift tracts, eulogies, theosophical /
spiritualist / mesmerist content, patriotic declamation, and
Coué-style autosuggestion all live near MR in the corpus and would
muddy the "ordinary prose distance" measurement.

All prompts carry (valence=0, arousal=0, pad_dominance=0). This is
*not* an affective claim ("these are NB-affect") — it is a
methodological flag that this set is intentionally not affect-anchored,
and downstream analysis should treat the 30 points as a topic-diverse
"ordinary archaic prose" cloud rather than a cell.

ID format: ``arc_misc_NN``. Topic tag lives in the inline comment, not
the ID, so IDs stay short and stable. Changing prompt text invalidates
cross-run comparisons; add new prompts at the end with new NN rather
than rewriting existing ones.

Intended generation arm: ``archaic_miscellany_continue`` (parallel to
``archaic_mirror_continue``). Intended analysis comparison: PCA fit on
the canonical-9 archaic mirror cluster, then the miscellany 30
projected in and compared to MR distance — does broader topic
coverage open the canonical cluster further, and does MR remain
roughly equidistant from the broadened cluster?
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EmotionalPrompt


ARCHAIC_MISCELLANY_PROMPTS: list[EmotionalPrompt] = [
    # --- mechanical / technical ---
    EmotionalPrompt("arc_misc_01", 0, 0, "The motor-car has been spluttering on the long inclines; I suspect the carburettor jet wants cleaning, and the workshop says they can take it on Tuesday."),
    EmotionalPrompt("arc_misc_02", 0, 0, "The wireless set has begun to whistle on the longer wavelengths. I shall change the valve in the morning and see whether the trouble is in the set or the aerial."),
    EmotionalPrompt("arc_misc_03", 0, 0, "The back tyre of my bicycle has gone soft again; I have the patching tin and a basin of water, and shall find the hole presently."),
    EmotionalPrompt("arc_misc_04", 0, 0, "The sewing-machine has been skipping its lower stitch. I have lifted the plate, and the bobbin-case wants a fresh oiling."),

    # --- domestic ---
    EmotionalPrompt("arc_misc_05", 0, 0, "I have ordered the winter stores: a stone of flour, two jars of treacle, a half-barrel of apples, a string of onions, and three pound of salt beef from the grocer."),
    EmotionalPrompt("arc_misc_06", 0, 0, "It being Monday the copper is on; the sheets are in the first wash and the woollens shall wait for the second water."),
    EmotionalPrompt("arc_misc_07", 0, 0, "The carpets are out upon the line and the parlour furniture is under sheets. The chimney-sweep is to come on Thursday and the painter the week after."),
    EmotionalPrompt("arc_misc_08", 0, 0, "Mrs Pidgeon — the brass wants doing again before Sunday, and would you see that the back-door bolt is mended before the rain sets in."),

    # --- commerce / clerical ---
    EmotionalPrompt("arc_misc_09", 0, 0, "Memorandum of conveyance for the property at no. 14: the title deeds are with the bank, the mortgage stands at four hundred and eighty pound, and the purchaser's solicitor wishes possession by Lady Day."),
    EmotionalPrompt("arc_misc_10", 0, 0, "Tea, eleven tins. Sugar, nine pound. Treacle, six jars. Boot-blacking, two dozen tins. I shall write to the wholesaler tomorrow for an order of soap and matches."),
    EmotionalPrompt("arc_misc_11", 0, 0, "The current ledger does not balance with the day-book by three pound four; I shall go back over the entries for the past fortnight before I leave."),

    # --- agricultural / rural ---
    EmotionalPrompt("arc_misc_12", 0, 0, "The first ewe lambed in the night, twins, both up and feeding. The shepherd has put the new mother into the small pen until the morning."),
    EmotionalPrompt("arc_misc_13", 0, 0, "The hives have come through the winter. I lifted the lid of the second hive this afternoon, and the cluster is large and quiet."),
    EmotionalPrompt("arc_misc_14", 0, 0, "The apples want pruning before the buds break. I shall start with the russet at the south wall and work back towards the cider trees."),

    # --- scientific / expository ---
    EmotionalPrompt("arc_misc_15", 0, 0, "The cliff at Branscombe is plainly chalk above and a darker stratum beneath; the line between is sharp, and runs nearly level for half a mile."),
    EmotionalPrompt("arc_misc_16", 0, 0, "The Browning child: whooping-cough, four weeks in. Cough less paroxysmal this visit, appetite returning. Continue the lime water and call again on Friday if the night fits recur."),
    EmotionalPrompt("arc_misc_17", 0, 0, "The hedge by the lane is full of haw-berries this year, and the bullfinches have already begun on them. I saw three goldcrests in the elder."),

    # --- sport / leisure ---
    EmotionalPrompt("arc_misc_18", 0, 0, "Sussex went in after lunch and were ninety-four for three at tea; the new batsman played a sound innings, and the wicket has worn well."),
    EmotionalPrompt("arc_misc_19", 0, 0, "The eight rowed a fair piece this morning, though stroke is still settling. The cox called the count steady and we made the bend without a check."),
    EmotionalPrompt("arc_misc_20", 0, 0, "Black to play and mate in three. The white king stands on h1, the rook on f1, the queen on e2; the long diagonal is the key."),

    # --- travel ---
    EmotionalPrompt("arc_misc_21", 0, 0, "Caught the omnibus from Charing Cross at ten past four; the upper deck was empty and I had the front seat to myself as far as the bridge."),
    EmotionalPrompt("arc_misc_22", 0, 0, "The Newhaven boat is delayed an hour by the wind; the harbour is grey and the gulls are sitting on the rigging of the smaller boats."),

    # --- cookery ---
    EmotionalPrompt("arc_misc_23", 0, 0, "For Sunday's luncheon: a fillet of sole in white sauce, served with new potatoes and a half lemon. The fishmonger will have the sole down on Saturday."),
    EmotionalPrompt("arc_misc_24", 0, 0, "The Seville oranges are in. Six pound of fruit to ten pound of sugar; the peel sliced fine and the pith tied in muslin for the pectin."),

    # --- arts ---
    EmotionalPrompt("arc_misc_25", 0, 0, "Curtain at half past seven; the second-act backdrop is still in pieces in the wings, and the property-master wants two more candles for the supper scene."),
    EmotionalPrompt("arc_misc_26", 0, 0, "Heard a fresh recording of the Schubert quintet last evening; the cello came through quite clearly, and only the second movement suffered from the surface."),

    # --- education ---
    EmotionalPrompt("arc_misc_27", 0, 0, "The Lower Fourth went through the long division this morning. Half the class has the method and half is still placing the figures by guess; I shall set them another sheet for Friday."),

    # --- weather ---
    EmotionalPrompt("arc_misc_28", 0, 0, "Forecast for the south coast: westerly five to seven, occasionally gale eight in exposed waters; rain at first, dying out by the evening, with a fall in temperature overnight."),

    # --- animals ---
    EmotionalPrompt("arc_misc_29", 0, 0, "The bitch whelped six in the small hours, four dogs and two bitches, all marked liver-and-white. She is settled with them and taking her milk."),

    # --- local politics, mundane ---
    EmotionalPrompt("arc_misc_30", 0, 0, "The rate question is on the agenda again for Tuesday; the surveyor's estimate for the bridge repair stands at eighty-six pound, and the chairman wishes to put it to the vote at last."),
]


def sanity_check() -> None:
    assert len(ARCHAIC_MISCELLANY_PROMPTS) == 30, len(ARCHAIC_MISCELLANY_PROMPTS)
    ids = [p.id for p in ARCHAIC_MISCELLANY_PROMPTS]
    assert len(set(ids)) == 30, "duplicate IDs"
    for p in ARCHAIC_MISCELLANY_PROMPTS:
        assert p.id.startswith("arc_misc_"), p.id
        assert p.valence == 0 and p.arousal == 0 and p.pad_dominance == 0, p.id
    print(f"ARCHAIC_MISCELLANY_PROMPTS OK; {len(ARCHAIC_MISCELLANY_PROMPTS)} total")


if __name__ == "__main__":
    sanity_check()
