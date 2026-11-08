"""Which support team handles which intent.

CLINC150 groups its 150 intents into 10 domains (15 intents each). Each domain is
treated as one support team. The 5 hand-written appliance intents go to the
"Appliance Service" team. Out-of-scope or low-confidence messages go to a human agent.
"""

DOMAINS = {
    "Banking": [
        "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance",
        "bill_due", "interest_rate", "routing", "min_payment", "order_checks", "pin_change",
        "report_fraud", "account_blocked", "spending_history",
    ],
    "Credit Cards": [
        "credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
        "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
        "credit_limit_change", "damaged_card", "replacement_card_duration",
        "improve_credit_score", "expiration_date",
    ],
    "Kitchen & Dining": [
        "recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
        "ingredients_list", "ingredient_substitution", "cook_time", "food_last",
        "meal_suggestion", "restaurant_reservation", "confirm_reservation", "how_busy",
        "cancel_reservation", "accept_reservations",
    ],
    "Home & Smart Devices": [
        "shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
        "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song", "order",
        "order_status", "reminder", "reminder_update", "smart_home",
    ],
    "Auto & Commute": [
        "traffic", "directions", "gas", "gas_type", "distance", "current_location", "mpg",
        "oil_change_when", "oil_change_how", "jump_start", "uber", "schedule_maintenance",
        "last_maintenance", "tire_pressure", "tire_change",
    ],
    "Travel": [
        "book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
        "travel_notification", "carry_on", "timezone", "vaccines", "translate",
        "flight_status", "international_visa", "lost_luggage", "plug_type", "exchange_rate",
    ],
    "Utilities": [
        "time", "alarm", "share_location", "find_phone", "weather", "text", "spelling",
        "make_call", "timer", "date", "calculator", "measurement_conversion", "flip_coin",
        "roll_dice", "definition",
    ],
    "HR & Work": [
        "direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance",
        "pto_request_status", "next_holiday", "insurance", "insurance_change",
        "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income",
    ],
    "Small Talk": [
        "greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
        "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you",
        "what_are_your_hobbies", "do_you_have_pets", "are_you_a_bot", "meaning_of_life",
        "who_do_you_work_for", "fun_fact",
    ],
    "Account Settings": [
        "change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
        "whisper_mode", "repeat", "no", "yes", "maybe", "change_language", "change_accent",
        "change_volume", "change_speed", "sync_device",
    ],
    "Appliance Service": [
        "dishwasher_issue", "washer_issue", "fridge_issue", "oven_issue", "warranty_claim",
    ],
}

OOS = "oos"
HUMAN_AGENT = "Human Agent"

INTENT_TO_TEAM = {intent: team for team, intents in DOMAINS.items() for intent in intents}
INTENT_TO_TEAM[OOS] = HUMAN_AGENT


def team_for(intent: str) -> str:
    return INTENT_TO_TEAM.get(intent, HUMAN_AGENT)
