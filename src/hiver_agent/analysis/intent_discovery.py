"""
Intent discovery engine for AmazonHelp customer support conversations.
Discovers empirical intent distributions using keyword pattern rules and text frequency analysis.
"""
import re
from typing import Dict, List, Tuple, Any
from collections import Counter
import pandas as pd


# 17 Empirically Grounded Intent Definitions & Pattern Matchers
INTENT_PATTERNS = {
    "order_tracking_delivery_status": [
        r'\bwhere(s|\sis)?\s+(my|the)\s+order\b', r'\btracking\b', r'\bstatus of my order\b',
        r'\bwhen will (it|my order|package) arrive\b', r'\bdelivery date\b', r'\bshipment status\b'
    ],
    "late_delivery_shipping_delay": [
        r'\blate\b', r'\bdelay(ed)?\b', r'\bnot arrived yet\b', r'\bexpected (yesterday|today|by)\b',
        r'\bstill hasn\'?t arrived\b', r'\bpast due\b', r'\boverdue\b'
    ],
    "missing_package_delivered_not_received": [
        r'\bdelivered but not\b', r'\bsays delivered\b', r'\bmarked as delivered\b', r'\bhaven\'?t received\b',
        r'\bmissing package\b', r'\blost package\b', r'\bnever got (it|my)\b'
    ],
    "refund_status_request": [
        r'\brefund\b', r'\bmoney back\b', r'\breimburse\b', r'\bwhen will i get my money\b', r'\brefund status\b'
    ],
    "return_exchange_inquiry": [
        r'\breturn\b', r'\bexchange\b', r'\breturn label\b', r'\bdrop off\b', r'\bhow (do|can) i return\b'
    ],
    "order_cancellation_request": [
        r'\bcancel\b', r'\bcancellation\b', r'\bcancelled\b', r'\bstop (the|my) order\b'
    ],
    "damaged_defective_item": [
        r'\bdamage(d)?\b', r'\bbroken\b', r'\bdefective\b', r'\bsmashed\b', r'\bshattered\b', r'\bdoes not work\b'
    ],
    "wrong_item_received": [
        r'\bwrong (item|product|order)\b', r'\breceived something else\b', r'\bdifferent item\b', r'\bmissing item(s)?\b'
    ],
    "payment_billing_issue": [
        r'\bcharge(d)?\b', r'\bbill(ing)?\b', r'\bdouble charge\b', r'\bpayment\b', r'\bgift card\b', r'\bcredit card\b'
    ],
    "account_access_login": [
        r'\baccount\b', r'\blogin\b', r'\bpassword\b', r'\botp\b', r'\block(ed)?\b', r'\bsign in\b', r'\bverification\b'
    ],
    "prime_membership_subscription": [
        r'\bprime\b', r'\bmembership\b', r'\bsubscrib(e|tion)\b', r'\bprime fee\b', r'\bstudent prime\b'
    ],
    "digital_content_kindle_video": [
        r'\bkindle\b', r'\bprime video\b', r'\baudible\b', r'\bfire stick\b', r'\bdigital\b', r'\bebook\b', r'\bapp\b'
    ],
    "third_party_seller_issue": [
        r'\bthird party\b', r'\bseller\b', r'\bmarketplace\b', r'\bfulfilled by\b'
    ],
    "promotions_discounts_coupons": [
        r'\bpromo(tion)?\b', r'\bcoupon\b', r'\bdiscount\b', r'\bvoucher\b', r'\bcode not working\b'
    ],
    "address_delivery_instructions": [
        r'\baddress\b', r'\bdelivery location\b', r'\bdriver\b', r'\bchange address\b', r'\bwrong address\b'
    ],
    "customer_service_complaint": [
        r'\bterrible service\b', r'\bworst\b', r'\bmanager\b', r'\bsupervisor\b', r'\bcall me\b', r'\bphone number\b'
    ]
}


def classify_single_text_intent(text: str) -> str:
    """
    Classify a customer query text into one of the 17 intent categories.
    """
    if not isinstance(text, str) or not text.strip():
        return "general_inquiry_other"

    text_lower = text.lower()
    scores = {}

    for intent_name, patterns in INTENT_PATTERNS.items():
        score = 0
        for pat in patterns:
            if re.search(pat, text_lower):
                score += 1
        if score > 0:
            scores[intent_name] = score

    if not scores:
        return "general_inquiry_other"

    # Return highest scoring intent
    best_intent = max(scores.items(), key=lambda x: x[1])[0]
    return best_intent


def discover_amazonhelp_intents(df_threads: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Annotate AmazonHelp threads with discovered intent labels and return distribution metrics.
    """
    print("Classifying AmazonHelp conversation intents...", flush=True)
    
    intents = []
    for txt in df_threads['first_customer_text']:
        intents.append(classify_single_text_intent(txt))

    df_threads['discovered_intent'] = intents

    intent_counts = Counter(intents)
    total_convs = len(df_threads)

    distribution = []
    for intent_name, count in intent_counts.most_common():
        pct = round((count / total_convs) * 100, 2) if total_convs > 0 else 0.0
        distribution.append({
            "intent_name": intent_name,
            "count": count,
            "percentage": pct
        })

    summary = {
        "total_classified_conversations": total_convs,
        "unique_intents_found": len(intent_counts),
        "top_intent": intent_counts.most_common(1)[0] if intent_counts else ("none", 0),
        "distribution": distribution
    }

    return df_threads, summary
