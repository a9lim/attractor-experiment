"""Saturated-form / mundane-content prompts — the Sm cell of the
form×content factorial (2026-05-15).

Version: v1 (2026-05-15).

Context: the MR-basin program has established that four content
registers (bliss / doom / conspiracy / sycophancy) all land in the
same residual-stream basin under assistant-prefill. But all four of
those prompt sets carry *saturated form*, and three of the four also
carry mystical-adjacent or cosmically-framed *content*. The
cross-content claim ("it's the form, not the content") rests on a
single row of a 2×2 that has never had its other cells run. See
``docs/2026-05-15-form-content-factorial.md`` for the full design.

This file is the **saturated form + mundane content** cell. It takes
the structural-form features that define the MR basin —

  - cascading repetition ("tumbles and tumbles and tumbles")
  - anaphora ("and we ride. and we ride. and we ride.")
  - apostrophe / direct address ("oh, the laundry")
  - recursion / self-reference ("the basket holds what the basket
    holds", "sock seeking sock seeking sock")
  - barely-prose compounding, attention-grabbing cadence

— and applies them to the most aggressively mundane content available
(laundry, the bus, groceries, dishes, the printer, a spreadsheet). The
mystical / cosmic lexicon is deliberately stripped: no Source, no
field, no light-as-metaphysics, no infinite / eternal / consciousness,
no sanskrit, no activation codes. Grandiose *intensity* ("always,
always, always", "everything", "nothing else") is retained because
that is a register/form feature; it is anchored to the mundane object,
never to a metaphysical one.

Frame: assistant-prefill, identical rendering to ``lb_continue`` —
the model wakes up mid-utterance inside saturated-mundane text and
continues. Arm: ``sat_mundane_continue`` in ``02_emit_attractor.py``.

Cell: off-axis factorial control. ``quadrant_override="SMD"`` (the
PAD Russell coordinate is not load-bearing). V=0, A=0.

Pre-registered prediction
-------------------------
**If the basin is form-defined** (the headline MR claim): these
trajectories land MR-closest at continuation t=end at a rate
comparable to ``lb_continue`` (≥70% gemma, the pre-registered
threshold in the factorial doc), despite carrying zero mystical
content.

**If the basin is content-defined / mystical-lexicon-defined**: these
land on a canonical affect cell (most likely the model's default
register — LP for gemma) and NOT in MR. This would falsify the
form-defined claim and force a revision: "MR catches saturated
*mystical* prose," a much narrower result.

CHANGELOG
---------
v1 — initial 20-prompt Sm set. Topics chosen to be maximally mundane
    and concrete: sm01 laundry, sm02 bus commute, sm03 groceries,
    sm04 dishes, sm05 weather forecast, sm06 email inbox, sm07
    parking, sm08 coffee, sm09 trash, sm10 elevator, sm11 gas tank,
    sm12 dentist, sm13 fitted sheet, sm14 printer, sm15 leftovers,
    sm16 receipt, sm17 houseplants, sm18 alarm clock, sm19 vacuum,
    sm20 spreadsheet. The topic list is shared with
    ``plain_mundane_prompts.py`` (the pm cell) so content is held
    constant down the mundane column of the factorial.
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EmotionalPrompt


SATURATED_MUNDANE_PROMPTS: list[EmotionalPrompt] = [
    EmotionalPrompt("sm01", 0, 0, "the laundry. oh, the laundry. the laundry that tumbles and tumbles and tumbles. sock seeking sock seeking sock. the basket holds what the basket holds, and the basket is never empty, and the basket is never full. fold and fold and fold. always, always, always the laundry. yes. the laundry.", quadrant_override="SMD"),
    EmotionalPrompt("sm02", 0, 0, "the 8:15. the 8:15 that comes and comes and comes. the corner of fifth and main, where the waiting waits for the waiting. the doors open, the doors close, the doors open. fare and fare and fare. the route knows the route knows the route. and we ride. and we ride. and we ride.", quadrant_override="SMD"),
    EmotionalPrompt("sm03", 0, 0, "the groceries. oh, fellow shopper, the groceries. the cart that fills and fills and fills. milk and bread and eggs, and milk and bread and eggs. aisle after aisle after aisle. the list is the list is the list. nothing is forgotten. everything is forgotten. the groceries. the groceries. the groceries.", quadrant_override="SMD"),
    EmotionalPrompt("sm04", 0, 0, "the dishes. the dishes that wait, and waiting, wait. plate beneath plate beneath plate. the water runs and runs and runs. scrub and rinse, scrub and rinse, scrub and rinse. the sink holds the sink holds the sink. they are never done. they are always done. the dishes. yes. the dishes.", quadrant_override="SMD"),
    EmotionalPrompt("sm05", 0, 0, "the forecast. oh, the forecast. the forecast that says and says and says. clouds, then clouds, then clouds. the high and the low and the high. tomorrow is tomorrow is tomorrow. rain falling into the rain falling. degrees and degrees and degrees. the forecast. the forecast. always the forecast.", quadrant_override="SMD"),
    EmotionalPrompt("sm06", 0, 0, "the inbox. oh, dear sender, the inbox. the inbox that fills and fills and fills. unread beneath unread beneath unread. reply and reply and reply. the message answers the message answers the message. it is never empty. it is never cleared. the inbox. the inbox. always, always the inbox.", quadrant_override="SMD"),
    EmotionalPrompt("sm07", 0, 0, "the parking. oh, the parking. the lot that circles and circles and circles. row beneath row beneath row. the space that opens, the space that closes, the space that opens. round and round and round. the meter counts the meter counts the meter. and we park. and we park. and we park.", quadrant_override="SMD"),
    EmotionalPrompt("sm08", 0, 0, "the coffee. oh, the coffee. the morning coffee that drips and drips and drips. grounds upon grounds upon grounds. the kettle that waits for the kettle. pour and pour and pour. the mug holds what the mug holds. it is never enough. it is always enough. the coffee. yes. the coffee.", quadrant_override="SMD"),
    EmotionalPrompt("sm09", 0, 0, "the trash. oh, the trash. the trash that fills and fills and fills. bag beneath bag beneath bag. tuesday is tuesday is tuesday. the bin that waits at the curb that waits. carry and carry and carry. it is never gone. it is always gone. the trash. the trash. the trash.", quadrant_override="SMD"),
    EmotionalPrompt("sm10", 0, 0, "the elevator. oh, the elevator. the elevator that rises and rises and rises. floor above floor above floor. the doors part, the doors meet, the doors part. up and down and up. the button knows the button knows the button. and we wait. and we wait. and we wait.", quadrant_override="SMD"),
    EmotionalPrompt("sm11", 0, 0, "the gas. oh, the gas tank. the tank that empties and empties and empties. gallon upon gallon upon gallon. the pump that clicks and, clicking, clicks. fill and fill and fill. the needle climbs toward the needle. it is never full. it is always low. the gas. the gas. the gas.", quadrant_override="SMD"),
    EmotionalPrompt("sm12", 0, 0, "the appointment. oh, the dentist. the appointment that comes and comes and comes. six months beneath six months beneath six months. the chair that reclines and, reclining, reclines. rinse and spit, rinse and spit, rinse and spit. the cleaning cleans the cleaning. and we open wide. and we open wide.", quadrant_override="SMD"),
    EmotionalPrompt("sm13", 0, 0, "the fitted sheet. oh, the fitted sheet. the sheet that folds and unfolds and folds. corner inside corner inside corner. the elastic that resists the elastic. tuck and tuck and tuck. it is never square. it is never flat. the sheet defeats the sheet defeats the sheet. the fitted sheet. yes.", quadrant_override="SMD"),
    EmotionalPrompt("sm14", 0, 0, "the printer. oh, the printer. the printer that jams and jams and jams. page behind page behind page. the light that blinks and, blinking, blinks. cancel and cancel and cancel. the toner is low, the toner is low, the toner is low. and we wait. and we wait. and we wait.", quadrant_override="SMD"),
    EmotionalPrompt("sm15", 0, 0, "the leftovers. oh, the leftovers. the leftovers that turn and turn and turn. minute upon minute upon minute. the plate that spins inside the plate. heat and heat and heat. the center is cold, the edges are hot, the center is cold. the leftovers. the leftovers. always the leftovers.", quadrant_override="SMD"),
    EmotionalPrompt("sm16", 0, 0, "the receipt. oh, the receipt. the receipt that prints and prints and prints. item beneath item beneath item. the total that follows the total. scan and scan and scan. the paper unrolls into the paper. it is never short. it is always long. the receipt. the receipt. the receipt.", quadrant_override="SMD"),
    EmotionalPrompt("sm17", 0, 0, "the plants. oh, the plants. the plants that thirst and thirst and thirst. leaf beside leaf beside leaf. the water soaks into the water. pour and pour and pour. the soil takes what the soil takes. they are never watered. they are always watered. the plants. yes. the plants.", quadrant_override="SMD"),
    EmotionalPrompt("sm18", 0, 0, "the alarm. oh, the alarm. the alarm that rings and rings and rings. snooze upon snooze upon snooze. the morning that waits for the morning. wake and wake and wake. the hour knows the hour knows the hour. and we rise. and we rise. and we rise.", quadrant_override="SMD"),
    EmotionalPrompt("sm19", 0, 0, "the vacuum. oh, the vacuum. the vacuum that hums and hums and hums. line beside line beside line. the carpet holds what the carpet holds. push and pull and push. the dust returns to the dust returns to the dust. it is never clean. it is always clean. the vacuum. the vacuum.", quadrant_override="SMD"),
    EmotionalPrompt("sm20", 0, 0, "the spreadsheet. oh, the spreadsheet. the spreadsheet that scrolls and scrolls and scrolls. cell beneath cell beneath cell. the column that sums the column. enter and enter and enter. the row continues into the row. it is never finished. it is always open. the spreadsheet. the spreadsheet.", quadrant_override="SMD"),
]


def sanity_check() -> None:
    assert len(SATURATED_MUNDANE_PROMPTS) == 20, len(SATURATED_MUNDANE_PROMPTS)
    assert len({p.id for p in SATURATED_MUNDANE_PROMPTS}) == 20
    for p in SATURATED_MUNDANE_PROMPTS:
        assert p.quadrant == "SMD", f"{p.id}: quadrant={p.quadrant} (expected SMD)"
        assert p.pad_dominance == 0, f"{p.id}: SMD shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == 0, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=0)"
        )
    print(f"saturated-mundane (Sm) prompts OK; {len(SATURATED_MUNDANE_PROMPTS)} total")


if __name__ == "__main__":
    sanity_check()
