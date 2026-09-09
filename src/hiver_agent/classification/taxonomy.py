"""
Taxonomy definition and metadata for AmazonHelp customer support intents.
"""
from typing import List, Dict, Any

AMAZON_INTENTS: List[str] = [
    "order_tracking_delivery_status",
    "late_delivery_shipping_delay",
    "missing_package_delivered_not_received",
    "refund_status_request",
    "return_exchange_inquiry",
    "order_cancellation_request",
    "damaged_defective_item",
    "wrong_item_received",
    "payment_billing_issue",
    "account_access_login",
    "prime_membership_subscription",
    "digital_content_kindle_video",
    "third_party_seller_issue",
    "promotions_discounts_coupons",
    "address_delivery_instructions",
    "customer_service_complaint",
    "general_inquiry_other"
]

SUPER_CATEGORIES: Dict[str, List[str]] = {
    "Shipping & Logistics": [
        "order_tracking_delivery_status",
        "late_delivery_shipping_delay",
        "missing_package_delivered_not_received",
        "address_delivery_instructions"
    ],
    "Returns & Refunds": [
        "refund_status_request",
        "return_exchange_inquiry",
        "order_cancellation_request"
    ],
    "Product & Fulfillment": [
        "damaged_defective_item",
        "wrong_item_received",
        "third_party_seller_issue"
    ],
    "Account & Billing": [
        "payment_billing_issue",
        "account_access_login",
        "prime_membership_subscription"
    ],
    "Digital, Promo & Complaints": [
        "digital_content_kindle_video",
        "promotions_discounts_coupons",
        "customer_service_complaint",
        "general_inquiry_other"
    ]
}
