from src.common import LABELS

DEPARTMENTS = dict(zip(LABELS, [
    "Billing Support", "Refund Operations", "Technical Helpdesk", "Identity & Access Support",
    "Logistics Support", "Membership Services", "Customer Care", "Product Advisory",
], strict=True))


def destination(label, confidence, threshold, departments=None):
    if not 0 <= confidence <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Confidence and threshold must be in [0, 1]")
    mapping = DEPARTMENTS if departments is None else departments
    return mapping[label] if confidence >= threshold else "Human Review Queue"
