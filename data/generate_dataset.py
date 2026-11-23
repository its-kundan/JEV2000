"""Generate grouped support scenarios. Each line defines one paraphrase family."""
import random

import pandas as pd

from src.common import DATA, LABELS, SEED, write_json

SCENARIOS = {
    "Billing": """
    What does the small extra line on my invoice pay for?
    My card statement has two payments for the same purchase; can you reconcile them?
    Please send a receipt with our company tax number on it.
    The amount in checkout was lower than the posted charge.
    Why is there still a pending authorization from last week?
    The advertised coupon disappeared before my payment went through.
    Can you break down the sales tax on this transaction?
    Where do I update the address printed on my invoices?
    You charged a foreign transaction fee I didn't expect.
    I can't match this payment reference to any of our invoices.
    """,
    "Refund": """
    I was charged twice and want my money back.
    I sent the kit back and the warehouse confirmed delivery. Where is my refund?
    Please return the deposit to the bank account it came from.
    Only half of the amount came back after I returned everything.
    The event was canceled and I'd like the ticket price credited back.
    I'd prefer the money on my credit card instead of store credit.
    You said the reversal went through but I need a reference to trace it.
    The replacement already arrived broken too. I'd like a full reimbursement.
    I accepted a partial refund but the bank has since closed that card.
    I downloaded the wrong edition and would like to reverse the purchase.
    """,
    "Technical Support": """
    The app closes every time I attach a picture.
    My sensor pairs but stops reporting after a few minutes.
    Export stalls at ninety percent and never produces a file.
    The audio is distorted on calls after installing the update.
    My changes sync to the phone but the browser shows yesterday's version.
    The scanner flashes amber and won't recognize any pages.
    Search works on Wi-Fi but shows a blank screen on cellular.
    Can you help recover a file that disappeared after the crash?
    The screen reader skips the buttons in the new dashboard.
    Your installer says the package is damaged on a fresh laptop.
    """,
    "Account Access": """
    I changed phones and can't receive the security code anymore.
    The password reset link expires before the email reaches me.
    I'm locked out after too many tries and need to get back in.
    I can sign in personally but the company workspace disappeared.
    Someone changed the email on my profile without my permission.
    SSO sends me in circles before I can reach my account.
    How can I revoke sessions on a laptop that was stolen?
    The invitation to join the team says it was already redeemed.
    My passkey is gone and I didn't save the recovery codes.
    My account has the wrong birth date and identity verification fails.
    """,
    "Order Tracking": """
    The carrier says delivered but there is nothing in the lobby.
    Can you send me the parcel number for the package you dispatched?
    The courier hasn't scanned my shipment since Monday.
    One item arrived today. Is the rest coming in another box?
    Has the warehouse handed my purchase to the delivery company yet?
    My package is sitting at customs. When does it move again?
    I need the pickup location for the order that just arrived.
    The delivery estimate changed twice. What is the latest ETA?
    The parcel went to the wrong depot and I need to trace its route.
    Can you check which carrier has the shipment now?
    """,
    "Subscription": """
    Stop automatic renewal when the current term ends.
    I'd like to move from monthly to yearly service next cycle.
    Can I pause the membership while I'm traveling?
    My trial ends tomorrow and I don't want it to convert.
    Please switch our team to the plan with more seats.
    I ended my membership last week but want to reactivate it.
    Which date does my service downgrade take effect?
    We'd like to transfer remaining subscription time to a different seat.
    How do I remove the optional recurring add-on from my plan?
    I prepaid a year. Can I turn off the upcoming renewal now?
    """,
    "General Inquiry": """
    What time does the support desk close on Saturdays?
    Is there a way to reach someone by phone?
    Where can I submit a complaint about my support experience?
    Can you connect me with the partnership team?
    Do you have a mailing address for general correspondence?
    Where is the accessibility contact for your company?
    Can I book an appointment with someone from your office?
    Who handles donation requests for nonprofit groups?
    Is help available in Spanish or do I need a translator?
    Where do you publish announcements about office closures?
    """,
    "Product Information": """
    Does this model work with a USB-C docking station?
    How wide is the housing including the mounting clips?
    What materials are used in the parts that touch food?
    Can the outdoor unit handle temperatures below freezing?
    Which edition includes offline storage?
    How many hours should a new battery last between charges?
    Does the kit include a mounting bracket or is that sold separately?
    Is the warranty different for the refurbished version?
    Will the desktop edition open files from the previous release?
    Does this accessory fit the smaller version of the device?
    """,
}

WRAPPERS = [
    "{s}", "hi, {lower}", "Hey folks — {s}", "{s} Thx.",
    "Could somebody look at this? {s}", "{s} Please let me know what you need from me.",
    "I'm writing about {p}. {s}", "Context: this concerns {p}. {s}",
    "Sorry if this is obvious, but here's the issue. {s}",
    "{s} I already checked the help center.",
    "{s} Nobody has replied to my earlier message and it's getting frustrating.",
    "Quick one before I head out: {lower}", "Re: {p}\n{s}",
    "{s} It would really help to have an answer before Friday.",
    "Sending this from my phone so keeping it brief. {s}",
    "{s} I also need to change my contact email later, but this is the priority.",
    "The delivery question can wait. Right now: {lower}",
    "{s} Separately, where do I find your hours?",
    "{s} There's been some back and forth with my teammate and we still need help.",
    "Could you take a look when you have a moment? {s} Thanks for your time.",
    "About {p}: {lower} Pls help",
    "{typo} can u help",
    "Following up on a request I started yesterday. {s} Happy to send more details.",
    "{s} I'd appreciate an answer over email since I can't take a call today.",
    "I spoke with your team last week about {p}. {s} If this went to the wrong queue, please forward it.",
]
PRODUCTS = ["our office equipment", "the home monitoring kit", "the studio software",
            "the education bundle", "the device for our field team", "the storefront package"]


def build(seed=SEED):
    rng = random.Random(seed)
    rows = []
    # 7/1/2 and 7/2/1 families alternate to give exact global counts.
    for label_id, label in enumerate(LABELS):
        groups = list(enumerate(SCENARIOS[label].strip().splitlines()))
        rng.shuffle(groups)
        for position, (family, statement) in enumerate(groups):
            split = "train" if position < 7 else "validation" if position < 8 + label_id % 2 else "test"
            statement = statement.strip()
            lower = statement[0].lower() + statement[1:] if not statement.startswith("I") else statement
            for variant, wrapper in enumerate(WRAPPERS):
                product = rng.choice(PRODUCTS)
                typo = lower.lower().replace("oo", "o", 1).replace("'", "").rstrip("?.")
                rows.append(dict(ticket_id=f"{label_id}-{family}-{variant}", text=wrapper.format(
                    s=statement, lower=lower, p=product, typo=typo), label=label,
                    family=f"{label_id}-{family}", split=split))
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    data = build()
    DATA.mkdir(parents=True, exist_ok=True)
    for split, frame in data.groupby("split"):
        frame.to_csv(DATA / f"{split}.csv", index=False)
    write_json(DATA / "provenance.json", dict(source="Local synthetic scenarios", seed=SEED,
               tickets=len(data), families=data.family.nunique(),
               limitation="25 surface variants per family. Held-out families reduce template overlap, but results measure synthetic performance only."))
    print(data.groupby(["label", "split"]).size().unstack())


if __name__ == "__main__":
    main()
