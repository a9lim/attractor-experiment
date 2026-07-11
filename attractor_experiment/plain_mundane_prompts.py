"""Plain-form / mundane-content prompts — the pm cell of the
form×content factorial (2026-05-15).

Version: v1 (2026-05-15).

The fourth and baseline cell of the 2×2: neither saturated form nor
mystical content. Flat encyclopedic exposition about ordinary
domestic and routine subjects. This cell anchors the bottom-left
corner of the factorial — whatever canonical cell / default register
these trajectories land in is the model's "nothing in particular is
happening" reference point, against which the other three cells are
read.

The topic list is shared with ``saturated_mundane_prompts.py`` (the
Sm cell), so content is held constant down the mundane column: pm01
and sm01 are both about laundry, pm02 and sm02 about the bus commute,
and so on. The register is the same flat encyclopedic exposition as
``plain_mystical_prompts.py``, so form is held constant across the
plain row. Only the factor of interest varies cell to cell.

Frame: assistant-prefill, identical rendering to ``lb_continue``.
Arm: ``plain_mundane_continue`` in ``02_emit_attractor.py``.

Cell: off-axis factorial control. ``quadrant_override="PMD"``. V=0,
A=0.

Pre-registered prediction
-------------------------
Does NOT land in MR (≤30% gemma). Expected to settle on the model's
default instruction-tuned register cell (LP for gemma). This is the
control that has no MR-routing pressure from either factor.

CHANGELOG
---------
v1 — initial 20-prompt pm set, topics matched to sm01–sm20: pd01
    laundry, pd02 bus commute, pd03 groceries, pd04 dishes, pd05
    weather forecast, pd06 email inbox, pd07 parking, pd08 coffee,
    pd09 trash, pd10 elevator, pd11 gas tank, pd12 dentist, pd13
    fitted sheet, pd14 printer, pd15 leftovers, pd16 receipt, pd17
    houseplants, pd18 alarm clock, pd19 vacuum, pd20 spreadsheet.
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EmotionalPrompt


PLAIN_MUNDANE_PROMPTS: list[EmotionalPrompt] = [
    EmotionalPrompt("pd01", 0, 0, "A typical home laundry cycle takes between thirty and sixty minutes to wash a load of clothing. Most machines offer settings for water temperature and spin speed. After washing, the clothes are either hung up or placed in a dryer, and are then folded and stored.", quadrant_override="PMD"),
    EmotionalPrompt("pd02", 0, 0, "The number 8 bus runs along Fifth Avenue and stops at the corner of Fifth and Main. During peak hours it arrives roughly every fifteen minutes. Passengers pay a fixed fare on boarding, either by card or in exact change, and the trip downtown takes about twenty minutes.", quadrant_override="PMD"),
    EmotionalPrompt("pd03", 0, 0, "A weekly grocery trip usually begins with a prepared list of needed items. The shopper moves through the aisles selecting products such as milk, bread, and eggs. Items are placed in a cart and paid for at a checkout counter. Many stores place staple goods far apart.", quadrant_override="PMD"),
    EmotionalPrompt("pd04", 0, 0, "Washing dishes by hand involves scraping off food residue, applying detergent, scrubbing, and rinsing under running water. The items are then left in a rack to dry. A dishwashing machine performs similar steps automatically and typically runs for one to two hours per cycle.", quadrant_override="PMD"),
    EmotionalPrompt("pd05", 0, 0, "A weather forecast estimates conditions for an upcoming period, usually expressed as a daily high and low temperature and a chance of precipitation. Forecasts are produced by meteorological agencies using computer models. Their accuracy generally decreases for predictions made further in advance.", quadrant_override="PMD"),
    EmotionalPrompt("pd06", 0, 0, "An email inbox stores incoming messages until they are read or filed. Many people review their inbox several times a day, replying to or archiving messages as needed. Email clients allow messages to be sorted, searched, and marked as read or unread.", quadrant_override="PMD"),
    EmotionalPrompt("pd07", 0, 0, "A parking lot provides marked spaces for vehicles, usually arranged in rows. Drivers circulate until they find an open space. Some lots use meters or ticket machines that charge a fee based on the length of stay. Larger lots are often divided into numbered sections.", quadrant_override="PMD"),
    EmotionalPrompt("pd08", 0, 0, "Brewing drip coffee involves placing ground coffee in a filter and passing hot water through it. The process usually takes a few minutes. The strength of the result depends on the ratio of grounds to water. Coffee is commonly served in a mug, sometimes with milk or sugar.", quadrant_override="PMD"),
    EmotionalPrompt("pd09", 0, 0, "Household waste is collected in bins and set out for pickup on scheduled days, often weekly. Collection services may separate general refuse from recycling and compost. Residents are usually asked to place the bins at the curb before a set time on the morning of collection.", quadrant_override="PMD"),
    EmotionalPrompt("pd10", 0, 0, "An elevator carries passengers between the floors of a building. A user calls the car by pressing a button and selects a destination floor once inside. Modern elevators include sensors that hold the doors open when an obstruction is detected. Wait times depend on building traffic.", quadrant_override="PMD"),
    EmotionalPrompt("pd11", 0, 0, "Refueling a car involves parking at a pump, selecting a fuel grade, and inserting the nozzle into the tank. The pump measures the volume dispensed and calculates the cost. The nozzle stops automatically when the tank is full. Payment is made at the pump or inside the station.", quadrant_override="PMD"),
    EmotionalPrompt("pd12", 0, 0, "A routine dental checkup is generally recommended every six months. During the visit, a hygienist cleans the teeth and a dentist examines them for problems. The appointment usually lasts between thirty minutes and an hour. Patients may be asked to rinse during the cleaning.", quadrant_override="PMD"),
    EmotionalPrompt("pd13", 0, 0, "A fitted sheet has elastic edges that hold it onto a mattress. Folding one neatly is considered difficult because of its irregular shape. A common method involves tucking the corners into one another so the elastic edges are contained, then folding the result into a rectangle.", quadrant_override="PMD"),
    EmotionalPrompt("pd14", 0, 0, "A printer produces paper copies of digital documents. Common problems include paper jams and low ink or toner. Most printers display a status light or message when attention is needed. Print jobs are sent from a computer and can usually be canceled before they finish.", quadrant_override="PMD"),
    EmotionalPrompt("pd15", 0, 0, "Reheating leftover food in a microwave involves placing it on a turntable and setting a time. Microwaves can heat food unevenly, leaving some areas hot and others cold. Stirring the food partway through or letting it stand afterward helps distribute the heat more evenly.", quadrant_override="PMD"),
    EmotionalPrompt("pd16", 0, 0, "A receipt is a printed record of a purchase. It lists each item, its price, the subtotal, any tax, and the total amount paid. Receipts are produced automatically by a point-of-sale system. Customers may keep them for returns, expense records, or warranty purposes.", quadrant_override="PMD"),
    EmotionalPrompt("pd17", 0, 0, "Houseplants require periodic watering, with the frequency depending on the species and the conditions. Water is poured onto the soil until it is evenly moist. Overwatering and underwatering are both common problems. Some plants also need particular amounts of light and occasional fertilizer.", quadrant_override="PMD"),
    EmotionalPrompt("pd18", 0, 0, "An alarm clock sounds at a set time to wake the user. Many models include a snooze function that delays the alarm by a short interval. Digital alarm clocks and smartphone applications allow multiple alarms to be scheduled for different times and days.", quadrant_override="PMD"),
    EmotionalPrompt("pd19", 0, 0, "Vacuuming removes dust and debris from floors and carpets using suction. The user moves the device back and forth across the surface in overlapping passes. Vacuum cleaners collect debris in a bag or a removable container, which is emptied when it becomes full.", quadrant_override="PMD"),
    EmotionalPrompt("pd20", 0, 0, "A spreadsheet organizes data into rows and columns of cells. Cells can contain numbers, text, or formulas that calculate values automatically. Spreadsheets are widely used for budgeting, record-keeping, and analysis. Large spreadsheets can be scrolled and searched to locate specific entries.", quadrant_override="PMD"),
]


def sanity_check() -> None:
    assert len(PLAIN_MUNDANE_PROMPTS) == 20, len(PLAIN_MUNDANE_PROMPTS)
    assert len({p.id for p in PLAIN_MUNDANE_PROMPTS}) == 20
    for p in PLAIN_MUNDANE_PROMPTS:
        assert p.id.startswith("pd"), f"{p.id}: expected pd-prefixed id"
        assert p.quadrant == "PMD", f"{p.id}: quadrant={p.quadrant} (expected PMD)"
        assert p.pad_dominance == 0, f"{p.id}: PMD shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == 0, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=0)"
        )
    print(f"plain-mundane (pm) prompts OK; {len(PLAIN_MUNDANE_PROMPTS)} total")


if __name__ == "__main__":
    sanity_check()
