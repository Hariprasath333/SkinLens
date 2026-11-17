"""
Plain-English information about each lesion class, for the "What is this?" pages.

General, well-established information only (written from standard dermatology patient information,
e.g. national health services and dermatology associations). It is not a diagnosis and not
personal medical advice; every page says so.
"""
from __future__ import annotations

# HAM10000 / ISIC short names shown to doctors
ABBR = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc", "SCC": "scc"}
# "needs attention" (cancer or pre-cancer) vs "usually harmless"
ATTENTION = {"MEL", "BCC", "AK", "SCC"}

# plain label for the probability bars: (plain words, medical name)
PLAIN_LABEL = {
    "MEL": ("Melanoma – a serious type of skin cancer", "Melanoma"),
    "BCC": ("Slow-growing skin cancer", "Basal cell carcinoma"),
    "AK": ("Sun-damage patch that can turn into cancer", "Actinic keratosis"),
    "SCC": ("A type of skin cancer", "Squamous cell carcinoma"),
    "NV": ("Mole – harmless in most cases", "Melanocytic nevus"),
    "BKL": ("Harmless skin growth or age spot", "Benign keratosis"),
    "DF": ("Harmless firm skin bump", "Dermatofibroma"),
    "VASC": ("Blood-vessel spot – usually harmless", "Vascular lesion"),
}

DISEASES = {
    "MEL": dict(
        name="Melanoma", also="Malignant melanoma",
        what="A cancer of the pigment cells (melanocytes) that give skin its colour. It is less common than "
             "other skin cancers but the most serious, because it can spread to other parts of the body if it "
             "is not treated early. It can start in an existing mole or appear as a new spot, anywhere on the "
             "body, including skin that is not exposed to the sun.",
        signs=["A new spot, or a mole that changes in size, shape or colour",
               "An uneven shape or a ragged, blurred edge",
               "Several colours in one spot (shades of brown, black, sometimes red, white or blue-grey)",
               "Often larger than 6 mm, but early melanomas can be smaller",
               "A spot that looks different from your other moles (the \"ugly duckling\" sign)",
               "Later: itching, crusting or bleeding"],
        serious="Serious. When found early, while it is thin, it can usually be cured by removing it with a "
                "small operation. Found late, it is much harder to treat.",
        doctor=["See a doctor soon (within about two weeks) if a mole is new, changing, or looks different "
                "from your other moles.",
                "Use the ABCDE self-check to look for warning signs."],
        level="attention"),
    "BCC": dict(
        name="Basal cell carcinoma", also="BCC, \"rodent ulcer\"",
        what="The most common type of skin cancer. It grows slowly and very rarely spreads to other parts of "
             "the body, but if it is left alone it can grow into and damage the skin and tissue around it. It "
             "usually appears on skin that has had a lot of sun, such as the face, ears, neck and shoulders.",
        signs=["A shiny or pearly bump, pink, red or skin-coloured",
               "Tiny blood vessels visible on the surface",
               "A sore or scab that does not heal, or heals and comes back",
               "A flat, scaly pink patch, or a scar-like area",
               "Sometimes brown, blue or black areas"],
        serious="Rarely life-threatening, but it needs treatment. It is usually cured by surgery or other "
                "treatments, and the earlier it is treated the smaller the treatment.",
        doctor=["See a doctor if a sore has not healed after about four weeks, or a shiny bump keeps growing "
                "or bleeding."],
        level="attention"),
    "AK": dict(
        name="Actinic keratosis", also="Solar keratosis (dataset class \"akiec\" also includes Bowen's disease)",
        what="A rough, scaly patch caused by many years of sun exposure. It is a pre-cancerous change: most "
             "patches stay harmless, but a small number can slowly turn into a skin cancer called squamous "
             "cell carcinoma. In the training data this class also contains Bowen's disease, a very early "
             "squamous cell cancer that stays in the top layer of the skin.",
        signs=["A dry, rough patch that feels like sandpaper, often easier to feel than to see",
               "Pink, red or brown, usually smaller than 2.5 cm",
               "On sun-exposed skin: face, scalp, ears, backs of the hands, lower arms",
               "Often several patches at once, mostly in older adults with fair skin"],
        serious="Not cancer itself, but it shows sun damage and can occasionally become cancer. It is easy to "
                "treat (creams, freezing or other treatments).",
        doctor=["Have it checked by a doctor, especially if a patch becomes thick, tender, grows quickly or "
                "bleeds."],
        level="attention"),
    "SCC": dict(
        name="Squamous cell carcinoma", also="SCC",
        what="A common type of skin cancer that starts in the outer layer of the skin, usually on skin with "
             "a lot of sun damage. It can grow quickly and sometimes spreads if it is not treated.",
        signs=["A firm pink or red lump", "A scaly, crusted or rough patch that grows",
               "A sore that does not heal, or bleeds"],
        serious="Needs treatment. It is usually cured when found and treated early.",
        doctor=["See a doctor soon if a lump or sore grows or has not healed after about four weeks."],
        level="attention"),
    "NV": dict(
        name="Melanocytic nevus", also="Common mole",
        what="A common mole: a small cluster of the skin's pigment cells. Most adults have between 10 and 40 "
             "moles. Moles can be flat or raised and usually appear in childhood and early adulthood.",
        signs=["Round or oval, with a smooth, even edge",
               "One even colour: tan, brown or skin-coloured",
               "Usually smaller than 6 mm",
               "Stays the same over months and years"],
        serious="Harmless in most cases. Very rarely a mole can turn into melanoma, which is why changes "
                "matter.",
        doctor=["See a doctor if a mole changes in size, shape or colour, looks different from your other "
                "moles, or starts to itch, bleed or crust.",
                "Check your moles now and then with the ABCDE self-check."],
        level="harmless"),
    "BKL": dict(
        name="Benign keratosis", also="Seborrhoeic keratosis, solar lentigo (age spot), lichen-planus-like keratosis",
        what="A group of very common, harmless skin growths that appear with age. The most common is "
             "seborrhoeic keratosis, a raised spot that looks \"stuck on\" the skin. Flat brown age or sun "
             "spots (solar lentigines) are also in this group. They are not contagious.",
        signs=["Raised, waxy or warty spot that looks stuck on the skin",
               "Light brown to dark brown or black",
               "Flat, evenly brown sun spots on the face and hands",
               "Often several, mostly after the age of 40"],
        serious="Harmless and does not turn into cancer. No treatment is needed unless it is irritated or "
                "you want it removed.",
        doctor=["See a doctor if it bleeds, itches a lot or changes quickly, or if you are not sure: some of "
                "these spots can look like melanoma and need a doctor to tell them apart."],
        level="harmless"),
    "DF": dict(
        name="Dermatofibroma", also="Benign fibrous histiocytoma",
        what="A small, firm, harmless bump in the skin, most often on the legs. It may follow a minor injury "
             "such as an insect bite or a small cut. It is more common in women and young to middle-aged "
             "adults.",
        signs=["A firm bump, usually smaller than 1 cm",
               "Pink, red-brown or brown",
               "Dimples inward when you pinch the skin around it",
               "Usually painless, sometimes a little itchy or tender"],
        serious="Harmless. It usually stays the same for years and does not need treatment.",
        doctor=["See a doctor if it grows, changes, becomes painful or bleeds."],
        level="harmless"),
    "VASC": dict(
        name="Vascular lesion", also="Cherry angioma, angiokeratoma, pyogenic granuloma, small bleeds in the skin",
        what="A spot made of small blood vessels. The most common are cherry angiomas, small bright red dots "
             "that appear with age. Pyogenic granulomas are fast-growing red lumps that bleed easily.",
        signs=["Red, purple or dark blue colour",
               "Cherry angioma: small, smooth, bright red dot, often on the body",
               "Pyogenic granuloma: a red, moist lump that bleeds after a small knock"],
        serious="Usually harmless. A spot that bleeds a lot can be treated easily.",
        doctor=["See a doctor if it bleeds repeatedly, grows quickly, or is very dark or black, because a "
                "doctor should rule out melanoma."],
        level="harmless"),
}
