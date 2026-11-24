"""Deterministic synthetic department → issue → regional product queue catalog."""
from itertools import product

from src.common import ROOT, write_json

CATALOG = ROOT / "data/generated/support_router/destinations.json"
REGIONS = ("APAC", "EMEA", "LATAM", "NA", "OCEANIA")
PRODUCTS = ("Atlas Cloud", "Beacon Mobile", "Cobalt Desk", "Drift Sync", "Ember Pay",
            "Forge API", "Grove Store", "Harbor Teams", "Ion Analytics", "Juniper Hub")
# Issue followed by six distinct ways a customer might describe it.
TAXONOMY = {
    "Billing": {
        "Payment failure": ("card payment keeps declining", "checkout won't accept my card", "transaction was rejected", "paymnt did not go through", "bank says the card is fine but I can't pay", "payment gateway rejects every attempt"),
        "Duplicate charge": ("charged twice for one order", "duplicate debit on my statement", "same payment appears two times", "u billed me twice", "there are two identical card charges", "my card was debited again for the same purchase"),
        "Invoice correction": ("wrong address on invoice", "please amend the billing receipt", "invoice has incorrect company details", "invioce needs a new name", "receipt shows my old business address", "the tax details on my bill need correcting"),
        "Payment method": ("update my saved credit card", "replace the payment method on file", "my card expires soon and needs swapping", "how do i add a new debit card", "switch the card used to pay", "remove an old card from the billing profile"),
        "Tax and fees": ("why is there an extra tax", "unexpected processing fee on my bill", "explain this sales tax charge", "what is this surcharge pls", "the listed price excluded a fee", "how was VAT calculated on my receipt"),
    },
    "Refund": {
        "Refund request": ("please issue a refund", "I want to get my money back", "return the amount I paid", "need a refnd asap", "can you reimburse this purchase", "I'd like the payment reversed"),
        "Refund status": ("where is my pending refund", "reimbursement still hasn't arrived", "when will the money be returned", "refund not showing yet", "I got approval but no credit", "track the repayment you started"),
        "Partial refund": ("only part of the payment was refunded", "the refund amount is too small", "missing half of my reimbursement", "got back less than i paid", "credit covers one item but not the rest", "the repayment only covered a fraction"),
        "Refund eligibility": ("am I eligible for a refund", "is this purchase refundable", "what is the money back policy", "can i still claim a refnd", "does the refund window still apply", "is reimbursement allowed on this item"),
        "Refund reversal": ("my refund credit was reversed", "the repayment appeared and then vanished", "refund was taken away again", "refnd disappeared from my statement", "the credited amount got debited again", "a reversal removed the reimbursement"),
    },
    "Technical Support": {
        "Application crash": ("app crashes as soon as it starts", "the program keeps closing unexpectedly", "screen freezes and then exits", "app keeps crashng", "launch ends in a blank window and shutdown", "opening the app causes it to quit"),
        "API error": ("API returns a 500 error", "requests to the endpoint keep failing", "integration receives invalid responses", "api call throws err", "the service endpoint responds with HTTP errors", "every webhook request fails at your API"),
        "Service interruption": ("the entire service is offline", "our workspace is down", "every user reports an outage", "service down for everyone rn", "none of our team can reach the service", "the platform has been unavailable all morning"),
        "Sync failure": ("my files are not syncing", "changes won't replicate between devices", "uploads show up on only one computer", "sync is stuck forever", "the desktop data never reaches my phone", "updates have stopped copying across devices"),
        "Performance degradation": ("everything takes ages to load", "app response times are very slow", "pages are timing out under light use", "super laggy today", "the interface hangs for several seconds", "dashboard latency has suddenly increased"),
    },
    "Account Access": {
        "Account lockout": ("my account is locked", "too many tries have blocked my profile", "please unlock my account", "locked out help", "access was disabled after failed attempts", "the sign-in screen says the user is suspended"),
        "Password reset": ("need a password reset", "forgot my password and can't change it", "reset link expired before I used it", "lost my pasword pls help", "send me a new password recovery link", "the reset email never arrived"),
        "Login failure": ("can't log in with my usual credentials", "sign-in keeps returning an error", "my username and password are rejected", "login fails every time", "I enter the correct details and it reloads", "authentication fails even with the right password"),
        "Two factor access": ("lost access to my authenticator", "verification code never arrives", "two factor challenge blocks my sign in", "new phone no 2fa codes", "MFA backup codes aren't accepted", "the one-time login token won't verify"),
        "Account recovery": ("recover my old account", "I can't access the email tied to my profile", "need to prove ownership and restore access", "lost my old email cant get in", "the recovery options point to an inactive inbox", "restore my account after losing the registered email"),
    },
    "Orders": {
        "Order delay": ("package is late", "the estimated delivery date has passed", "shipment hasn't moved", "order still not here", "parcel is overdue and tracking is stale", "the courier missed the scheduled delivery"),
        "Order tracking": ("where is my package", "need the shipment tracking number", "which carrier has my parcel", "got a tracking link pls", "how do I follow this delivery", "share the courier tracking details"),
        "Damaged shipment": ("the box arrived damaged", "item was broken in transit", "shipment showed up crushed", "parcel came busted", "the courier delivered a cracked product", "there was visible damage when I opened the package"),
        "Address change": ("change the shipping address", "the parcel is going to the wrong place", "I moved after placing the order", "ship to my new addr pls", "the courier needs a different delivery address", "update the destination before dispatch"),
        "Missing item": ("one item was missing from the box", "order arrived incomplete", "the parcel did not include everything", "box missing a thing i bought", "I received only part of my shipment", "packing slip lists something absent from the parcel"),
    },
    "Subscriptions": {
        "Cancellation": ("cancel my subscription", "stop my membership at the end of the month", "I no longer want this recurring plan", "pls cancel my sub", "close the subscription before it renews", "turn off my paid membership"),
        "Plan change": ("switch to a different subscription tier", "upgrade my monthly plan", "need to downgrade the membership", "wanna change plans", "can I move to the annual plan", "move my subscription to the smaller tier"),
        "Renewal": ("when does my subscription renew", "need help with membership renewal", "does the plan renew automatically", "is my sub renewing soon", "my paid plan is about to roll over", "confirm the next subscription renewal date"),
        "Trial extension": ("can you extend the trial", "need more time to evaluate the trial plan", "my free trial ends tomorrow", "extend my trial pls", "we haven't finished evaluating before the trial expires", "give us another week on the evaluation period"),
        "Seat management": ("add more seats to our plan", "remove an unused membership seat", "how many users does the plan include", "need 3 extra seats asap", "our team needs additional subscription licenses", "reduce our paid seat allocation"),
    },
    "Product Support": {
        "Feature guidance": ("how do I use this feature", "show me how the dashboard works", "need instructions for exporting reports", "how to use this tool pls", "where is the guide for this feature", "can you explain the reporting functionality"),
        "Compatibility": ("does this work on my device", "check compatibility with my operating system", "is my browser supported", "does this run on linux", "which versions are compatible with this product", "tell me the minimum system requirements"),
        "Product setup": ("help configure the product for the first time", "need the installation steps", "how do I set up the new workspace", "setup instructions pls", "walk me through onboarding this product", "what should I do after installing"),
        "Feature request": ("I'd like you to add a feature", "can you build a bulk export option", "suggestion for the product roadmap", "pls add dark mode", "we'd love support for custom dashboards", "I have an idea for a new product capability"),
        "Product comparison": ("compare these product versions", "what differs between the two editions", "help me choose between products", "whats the diff between models", "which edition includes the richer reports", "explain the differences in the product lineup"),
    },
    "General Support": {
        "Contact information": ("what is your support phone number", "how can I reach the help desk", "need your customer service email", "whats your contact info", "where do I find a support number", "which channels can I use to contact you"),
        "Business hours": ("when is your help desk open", "what are customer service hours", "are you available on the weekend", "r u open today", "when can I reach a representative", "what time does the support desk close"),
        "Feedback": ("I want to leave feedback", "where can I send a complaint", "I'd like to share my experience", "got feedback about ur service", "I'd like someone to log a complaint", "can I send a compliment about your team"),
        "Policy inquiry": ("where are your terms of service", "I have a question about the privacy policy", "send me your company policy", "where can i read ur terms", "I need the conditions that govern service use", "which policy applies to customer data"),
        "Unclear request": ("can someone help me with something", "not sure which team to contact", "I have a question but don't know where to start", "hi need help pls", "can I talk to an agent about an issue", "I need assistance and am unsure what to ask"),
    },
}


# Vocabulary a support team would use to describe a queue. The ticket generator
# uses the narrative paraphrases above, independently of this retrieval index.
QUEUE_KEYWORDS = {
    'Payment failure': ('decline', 'card rejected', 'checkout error'),
    'Duplicate charge': ('charged twice', 'double billing', 'duplicate debit'),
    'Invoice correction': ('billing receipt', 'amend invoice', 'incorrect address'),
    'Payment method': ('saved card', 'debit card', 'payment profile'),
    'Tax and fees': ('vat', 'surcharge', 'processing fee'),
    'Refund request': ('refund me', 'money back', 'return payment'),
    'Refund status': ('pending refund', 'refund tracking', 'missing credit'),
    'Partial refund': ('incomplete refund', 'partial reimbursement', 'refund amount'),
    'Refund eligibility': ('refund rules', 'money back policy', 'refund window'),
    'Refund reversal': ('reversed credit', 'vanished refund', 'refund clawback'),
    'Application crash': ('crashes', 'freeze', 'unexpected exit'),
    'API error': ('api request', 'endpoint', 'webhook'),
    'Service interruption': ('outage', 'service down', 'platform unavailable'),
    'Sync failure': ('replication', 'syncing', 'file synchronization'),
    'Performance degradation': ('slow', 'lag', 'latency'),
    'Account lockout': ('locked out', 'unlock', 'suspended profile'),
    'Password reset': ('forgot password', 'reset link', 'recovery email'),
    'Login failure': ("can't log in", 'invalid credentials', 'sign-in error'),
    'Two factor access': ('mfa', '2fa', 'authenticator'),
    'Account recovery': ('restore account', 'ownership', 'registered email'),
    'Order delay': ('shipment late', 'overdue', 'missed delivery'),
    'Order tracking': ('courier tracking', 'tracking number', 'carrier'),
    'Damaged shipment': ('broken item', 'damaged box', 'cracked parcel'),
    'Address change': ('shipping address', 'redirect parcel', 'new delivery address'),
    'Missing item': ('missing product', 'incomplete order', 'packing slip'),
    'Cancellation': ('cancel', 'stop renewal', 'end membership'),
    'Plan change': ('upgrade', 'downgrade', 'subscription tier'),
    'Renewal': ('auto-renewal', 'next renewal', 'billing cycle'),
    'Trial extension': ('extend trial', 'evaluation period', 'trial expiry'),
    'Seat management': ('licenses', 'membership seats', 'user allocation'),
    'Feature guidance': ('how to use', 'user guide', 'functionality'),
    'Compatibility': ('supported browser', 'system requirements', 'operating system'),
    'Product setup': ('install', 'initial configuration', 'onboarding'),
    'Feature request': ('suggestion', 'product roadmap', 'new capability'),
    'Product comparison': ('compare editions', 'differences', 'which version'),
    'Contact information': ('contact support', 'help desk number', 'customer service email'),
    'Business hours': ('opening hours', 'weekend availability', 'closing time'),
    'Feedback': ('complaint', 'compliment', 'customer experience'),
    'Policy inquiry': ('terms', 'privacy', 'customer data'),
    'Unclear request': ('unspecified issue', 'general help', 'talk to an agent'),
}


def build_destinations():
    rows = []
    for department, issues in TAXONOMY.items():
        for issue in issues:
            for region, name in product(REGIONS, PRODUCTS):
                slug = "-".join(issue.upper().split())
                rows.append(dict(
                    destination_id=f"{department.split()[0].upper()}-{slug}-{region}-{PRODUCTS.index(name)+1:03}",
                    department=department, subcategory=issue, team_name=f"{name} · {issue} · {region}",
                    description=f"{issue} support for {name} customers in {region}.",
                    supported_issue_types=[issue], region=region, product=name,
                    priority_support=region in {"APAC", "NA"}, routing_keywords=list(QUEUE_KEYWORDS[issue]),
                    active_status=True, synthetic=True,
                ))
    return rows


def main():
    rows = build_destinations()
    write_json(CATALOG, rows)
    print(f"Generated {len(rows):,} synthetic destinations in {CATALOG}")


if __name__ == "__main__":
    main()
