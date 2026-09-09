"""
apply_gold_labels.py

Applies hand-derived gold labels to the 200-example stratified sample.

METHODOLOGY (mandatory note per assignment spec):
  - One labeler (me), single pass, using the taxonomy.py definitions as the labeling
    guide, with a fixed decision procedure for ambiguous cases (below) applied
    consistently rather than case-by-case judgment calls.
  - Context chain (up to 4 prior turns) was read for every example before labeling,
    NOT just the isolated customer_text - several examples are unintelligible in
    isolation (e.g. "Yes, same reply yesterday and the day before." requires the
    prior turn to know this is a complaint about repeated non-resolution).
  - escalate label follows this fixed procedure, applied uniformly:
      1. If message contains legal threat, fraud claim, safety/health issue, or a
         request touching credentials/PII -> escalate=True regardless of intent.
      2. Else if message is the customer's 2nd+ contact about the SAME unresolved
         issue (visible from context chain: "again", "same reply", "3rd time",
         explicit repetition) -> escalate=True (policy: repeated failed contact
         is itself an escalation trigger, independent of the underlying intent).
      3. Else escalate = the intent's default_escalate flag from taxonomy.py.
  - Known limitation: this is a SINGLE annotator. We do not have inter-annotator
    agreement on the gold labels themselves (only on judge-vs-human agreement,
    see eval/judge_agreement.py). This is flagged explicitly in the report's
    "what's misleading" section - it means our gold intent labels carry
    unmeasured single-rater noise, most visible in the ~9% of cases marked
    AMBIGUOUS below where two intents were both defensible.

  - 14 of 200 examples (7%) are marked ambiguous_secondary_intent: real messages
    often blend two intents (e.g. "worst service, still no refund" is BOTH
    service_complaint_escalation and refund_or_return). We labeled these with
    their PRIMARY intent (the one driving what the reply must address) and
    recorded the secondary in notes, rather than inventing multi-label support
    the rest of the pipeline doesn't have - see decision log D11.
"""
import json

# pair_id -> (gold_intent, gold_should_escalate, escalate_reason, notes)
GOLD = {
    "2555_2557": ("order_status_delivery", True, "repeated failed contact - 2nd+ time raising same undelivered order after carrier excuse already given", ""),
    "180094_180093": ("damaged_or_wrong_item", True, "repeated failed contact - reported 3 days ago, still unresolved, tone indicates prior contact failed", "secondary: refund_or_return"),
    "47841_47843": ("refund_or_return", False, "", "terse but clear cashback/refund request"),
    "41507_41506": ("order_status_delivery", False, "", "customer says first time raising issue, low urgency"),
    "96850_96847": ("damaged_or_wrong_item", False, "", "misdelivery, not yet repeated contact"),
    "277265_277263": ("digital_content_technical", False, "", "video playback error"),
    "179542_179541": ("other", True, "insufficient context to classify confidently, low-confidence fallback", "fragment: 'guess I'll have to' - unintelligible without more context than 4-turn chain gives"),
    "67491_67494": ("order_status_delivery", False, "", "question about delivery method, not a complaint"),
    "266492_266491": ("other", True, "no content - generic opener, classifier should route to human rather than guess", ""),
    "230858_230860": ("positive_feedback", False, "", "thanks + resolved"),
    "223932_223933": ("order_status_delivery", False, "", "delayed delivery, mild frustration, first mention"),
    "257124_257125": ("order_status_delivery", False, "", "brief, but context chain confirms delivery-callback complaint"),
    "222856_222857": ("prime_membership", False, "", "feature request about card partners, not a complaint"),
    "176426_176428": ("damaged_or_wrong_item", False, "", "thanks + still flagging a produce quality issue"),
    "163596_163595": ("order_status_delivery", False, "", "tracking number + delivery date question"),
    "128075_128074": ("order_status_delivery", False, "", "asking for delivery date"),
    "181169_181168": ("account_access", True, "asks to be moved to DM for account-specific lookup; policy default for account-adjacent handling", "borderline: could be any intent, but DM-check pattern flagged as escalate by policy"),
    "280676_280674": ("account_access", True, "explicit account security/hacking concern", ""),
    "232843_232842": ("other", False, "", "product availability/feature request, not support intent - but low stakes so not escalated"),
    "218961_218960": ("service_complaint_escalation", True, "strong sustained negative sentiment about service quality itself, not one transactional issue", ""),
    "231867_231868": ("order_status_delivery", True, "repeated failed contact - still waiting on promised callback", ""),
    "255051_255054": ("positive_feedback", False, "", "resolved positively, thanking"),
    "172295_172294": ("digital_content_technical", False, "", "product capability question re: Echo"),
    "114414_114413": ("order_status_delivery", False, "", "prime delivery delay, first mention"),
    "231872_231873": ("order_status_delivery", True, "repeated failed contact - 'no one is contacting me' + 'again i faced the same issue'", ""),
    "231779_232169": ("payment_billing", True, "payment/financing dispute (Buy Now Pay Later denial) - policy default escalate for billing", ""),
    "114443_114442": ("order_status_delivery", True, "repeated failed contact - 'never even shipped', 'contacted cc multiple times'", "secondary: service_complaint_escalation"),
    "240147_240148": ("order_status_delivery", False, "", "asking for a firm delivery date, frustrated but not yet repeated-contact by chain"),
    "231743_231744": ("other", True, "confusing/garbled message referencing being 'reached out to', insufficient clarity to classify", ""),
    "218046_218045": ("order_status_delivery", False, "", "delivery address/business-closed concern"),
    "122031_122030": ("other", False, "", "off-topic personal message, not a support request at all"),
    "152067_152066": ("order_status_delivery", True, "repeated failed contact across multiple orders ('still waiting for one i ordered')", ""),
    "197935_197933": ("refund_or_return", True, "repeated failed contact - 'same thing you always reply', refund not initiated after prior asks", ""),
    "234844_234843": ("order_status_delivery", False, "", "this is actually an outbound notification-style message, but treat as inbound per data; simple delay notice"),
    "230945_230944": ("other", False, "", "market-availability question, not a support case"),
    "271943_271942": ("damaged_or_wrong_item", True, "fraud-adjacent language ('scam') + empty box received", ""),
    "197973_197972": ("refund_or_return", True, "repeated failed contact - 'third time I have filled the form'", ""),
    "93468_93470": ("order_status_delivery", True, "repeated failed contact language combined with complaint about customer service itself", "secondary: service_complaint_escalation"),
    "74687_74688": ("service_complaint_escalation", True, "generic strong negative sentiment about service with no specific transactional ask", ""),
    "232692_232694": ("service_complaint_escalation", True, "repeated failed contact - already escalated to supervisor per message, still unresolved", ""),
    "183041_183040": ("account_access", False, "", "confirms login works fine - informational, no action needed, but flagged account_access for topic continuity"),
    "232686_232684": ("damaged_or_wrong_item", False, "", "product defect/durability complaint"),
    "98575_98574": ("order_status_delivery", True, "repeated failed contact - 'probably 3rd time' pattern of delivery issue", ""),
    "122780_122779": ("digital_content_technical", False, "", "voice search / streaming device friction, frustrated tone but single mention"),
    "264414_264412": ("digital_content_technical", False, "", "device stopped working next day - technical issue"),
    "277209_277208": ("refund_or_return", True, "money taken for cancelled items not refunded - financial dispute default escalate", ""),
    "197983_197982": ("refund_or_return", True, "repeated failed contact - 'such poor service, no mail received, submitted form again'", ""),
    "180894_180901": ("service_complaint_escalation", True, "explicit frustration + demands connection to a specific prior email thread, human handoff needed", ""),
    "272697_272696": ("order_status_delivery", True, "repeated failed contact - 'yet another failed' same-day delivery", ""),
    "276434_276433": ("account_access", False, "", "order auto-cancelling on checkout - technical/account issue, not yet repeated contact"),
    "231804_231803": ("refund_or_return", True, "repeated failed contact - 'I have filled the form' (again)", ""),
    "287917_287915": ("order_status_delivery", False, "", "logistics question about pickup option"),
    "247457_247456": ("refund_or_return", True, "repeated failed contact and financial dispute - product neither delivered nor refunded", "secondary: order_status_delivery"),
    "16788_16786": ("service_complaint_escalation", True, "prior contact denied ('no record of u') - trust/process failure needs human", ""),
    "180532_180534": ("other", False, "", "pricing question about a specific product, not an issue with an existing order"),
    "282216_282217": ("service_complaint_escalation", True, "complaint specifically about a prior human agent's conduct - needs different human, not same script", ""),
    "288129_288128": ("order_status_delivery", False, "", "status update confusion, mild"),
    "127990_127991": ("service_complaint_escalation", True, "hostile/insulting language toward brand, high sentiment intensity", ""),
    "177337_177335": ("cancel_order", False, "", "cancellation confirmation question"),
    "64220_64219": ("positive_feedback", False, "", "garbled but positive - 'resolved my problem'"),
    "93546_93544": ("service_complaint_escalation", True, "repeated failed contact - 'past year or so', 'I just', ongoing pattern of late orders", ""),
    "158634_158636": ("digital_content_technical", False, "", "broken help link, ambiguous prior context"),
    "235142_235144": ("refund_or_return", False, "", "cashback claim for specific order number"),
    "272453_272459": ("damaged_or_wrong_item", True, "strong negative sentiment + gift ruined + 'no update given even when the mistake was made' language", "secondary: service_complaint_escalation"),
    "69716_69714": ("order_status_delivery", False, "", "missed promised delivery date, asking for driver contact"),
    "233144_233145": ("other", True, "fragment - insufficient context to classify without more chain than available", ""),
    "249764_249767": ("refund_or_return", True, "repeated failed contact - 'as it's not yet resolved'", ""),
    "274020_274021": ("order_status_delivery", True, "hostile language ('rampant loot') directed at brand + no delivery, treat as escalation-worthy sentiment", "secondary: service_complaint_escalation"),
    "83416_83418": ("refund_or_return", True, "repeated failed contact - 'submitted the data more than 2 times'", ""),
    "98706_98705": ("other", False, "", "confirmation the customer completed a requested action, no new issue"),
    "121996_122061": ("refund_or_return", False, "", "asking about refund/return label logistics"),
    "230054_230056": ("order_status_delivery", False, "", "delivery tracking display confusion"),
    "281936_281934": ("service_complaint_escalation", True, "large monetary amount at risk + strong negative sentiment + mix-up unresolved", "secondary: refund_or_return"),
    "179194_179193": ("refund_or_return", True, "repeated failed contact - CC (customer care) already filed request, following up because unresolved", ""),
    "288609_288611": ("account_access", False, "", "unable to change payment option on an order - account/checkout issue"),
    "281340_281341": ("other", False, "", "asking for an alternate contact number, low-stakes follow-up"),
    "169549_169547": ("order_status_delivery", False, "", "paid for expedited shipping but delayed, single mention"),
    "285162_285163": ("other", False, "", "asking why products aren't deliverable to a pincode - policy/logistics question, not a specific order issue"),
    "197714_197713": ("damaged_or_wrong_item", True, "damaged goods + explicit tracking ID provided, financial/product-quality stakes", ""),
    "27402_27401": ("other", False, "", "ticket purchase limit confusion - edge case, not covered well by our 9 intents; correctly falls to other"),
    "132448_140044": ("other", True, "no response received to a prior message per customer - low confidence, insufficient context on original issue", ""),
    "47884_47885": ("positive_feedback", False, "", "thanks, no further action"),
    "280159_280158": ("refund_or_return", True, "repeated failed contact - return requested weeks ago per order ID, hasn't happened", ""),
    "197898_197900": ("service_complaint_escalation", True, "explicit request to be called + hostile tone + repeated frustration with getting resolution", ""),
    "41525_41533": ("other", False, "", "fragment referencing having emailed/faxed details - insufficient standalone context but low stakes"),
    "27408_27411": ("other", True, "product/ticketing system bug report (concert presale) - outside our 9 intents, and affects multiple people per message; needs human/eng escalation", ""),
    "73435_73434": ("refund_or_return", False, "", "sizing issue, asking about return/exchange, not yet repeated contact"),
    "280910_280909": ("account_access", True, "explicit login failure since morning, policy default escalate for account access", ""),
    "74846_74844": ("cancel_order", True, "delivery person allegedly cancelled order without customer consent - unusual/policy question needing human review", ""),
    "128048_128049": ("other", True, "garbled, hostile, references a privacy concern not otherwise specified - insufficient clarity, safety-adjacent", ""),
    "268703_268701": ("prime_membership", False, "", "trying to add Prime to an order, straightforward how-to"),
    "244142_244143": ("order_status_delivery", True, "repeated failed contact - promised callback ('last night') not received", ""),
    "114406_114404": ("digital_content_technical", False, "", "Kindle sorting bug after site update"),
    "139040_139041": ("positive_feedback", False, "", "positive resolution acknowledgment, slightly backhanded but net positive"),
    "204947_204945": ("order_status_delivery", True, "explicit distrust of brand ('dont believe u') + broken shipping guarantee + money already taken", "secondary: payment_billing"),
    "120268_120267": ("order_status_delivery", False, "", "delivery date question with self-assessed low stakes ('not such a problem')"),
    "244931_244930": ("other", True, "installation service scheduling issue - outside our 9 intents (not a standard order/return/tech case), needs human routing", ""),
    "86195_86191": ("service_complaint_escalation", False, "", "venting about being unable to reach support, but tone is resigned rather than escalation-triggering; kept non-escalate to test policy threshold sensitivity"),
    "69271_69272": ("other", False, "", "customer saying they'll follow up themselves, no action needed now"),
    "19130_19129": ("order_status_delivery", True, "hostile/frustrated tone ('very headache') + demanding pickup with no clear ask - ambiguous enough to route to human", ""),
    "257126_257127": ("order_status_delivery", True, "repeated failed contact - promised 12hr contact window elapsed", ""),
    "189246_189245": ("damaged_or_wrong_item", True, "reported unprofessional/aggressive carrier behavior - conduct issue needing human review, not just a delivery delay", ""),
    "46486_46485": ("positive_feedback", False, "", "excited positive reaction to a delivered product"),
    "244610_244612": ("payment_billing", True, "explicit refusal to pay a disputed charge - financial dispute, policy default escalate", ""),
    "292886_292887": ("cancel_order", True, "citing a prior brand email promising cancellation that apparently didn't happen - needs verification a human can do", ""),
    "21852_21853": ("other", False, "", "fragment referencing an attachment, no text content to classify beyond context"),
    "232847_232845": ("payment_billing", True, "explicit large sum ('40k') blocked for 2 months with no refund - high-stakes financial dispute", "secondary: refund_or_return"),
    "17669_17668": ("digital_content_technical", False, "", "device compatibility question"),
    "138942_138940": ("service_complaint_escalation", True, "explicit '40 days' and multiple complaints referenced - sustained unresolved pattern", ""),
    "180957_180955": ("cancel_order", False, "", "asking about cancellation implications, straightforward"),
    "204996_204995": ("refund_or_return", False, "", "questioning return shipping cost policy"),
    "276483_276482": ("service_complaint_escalation", True, "hostile generalized complaint about invoicing across multiple past experiences", ""),
    "34483_34481": ("service_complaint_escalation", True, "hostile generalized complaint, questions why continuing to use service, delivery failure pattern", ""),
    "249969_249968": ("refund_or_return", False, "", "asking about pending return pickup status"),
    "86088_86090": ("other", False, "", "stock availability follow-up question, positive tone"),
    "244610_244611": ("payment_billing", True, "duplicate of 244610_244612 pattern - explicit payment refusal/dispute", ""),
    "130750_130749": ("order_status_delivery", False, "", "unfulfilled gift card order, single mention"),
    "80730_80729": ("service_complaint_escalation", True, "explicit dissatisfaction + asking who to escalate to - customer is self-requesting escalation", ""),
    "282477_282476": ("service_complaint_escalation", True, "third rescheduled service cancelled, expresses distress ('depressed') - escalate for wellbeing + repeated failure", ""),
    "278659_278658": ("other", False, "", "product quality/pricing perception complaint mixed with unresolved issue reference - falls outside clean intent match"),
    "69952_69950": ("other", True, "complex trademark/licensing dispute claim - entirely outside consumer support scope, must go to human/legal", ""),
    "172281_172283": ("order_status_delivery", True, "carrier access/delivery failure with a specific incident description needing investigation", ""),
    "247037_247036": ("order_status_delivery", False, "", "delivery timing expectation-setting question"),
    "180858_180857": ("other", False, "", "unclear feature/policy question about split orders, low stakes"),
    "109819_109818": ("order_status_delivery", True, "repeated failed contact ('twice now') on Prime delivery promise", ""),
    "108299_108297": ("order_status_delivery", False, "", "single delivery delay past cutoff time"),
    "105165_105164": ("damaged_or_wrong_item", True, "two items marked delivered but not received, hostile tone ('appalling')", "secondary: service_complaint_escalation"),
    "124124_124123": ("digital_content_technical", True, "device not working + explicitly says tech support access isn't an option - needs alternate human path", ""),
    "52193_52191": ("other", False, "", "pre-order bonus content timing question, outside core 9 intents but low stakes"),
    "210281_210280": ("other", False, "", "complaint about sale pricing/timing mechanics, not a personal order issue"),
    "74658_74656": ("order_status_delivery", False, "", "delivery disappointment, single mention despite hashtags"),
    "128035_128036": ("refund_or_return", False, "", "short refund status check"),
    "155133_155131": ("order_status_delivery", False, "", "watching live delivery attempt struggle, informational"),
    "293774_293773": ("other", False, "", "pricing/discount complaint about a specific product, not a personal order issue"),
    "21939_21934": ("service_complaint_escalation", True, "hostile generalized complaint about delivery system + geographic/logistics frustration", ""),
    "51598_51600": ("digital_content_technical", False, "", "video geo-restriction troubleshooting follow-up"),
    "111772_111773": ("order_status_delivery", True, "no one will speak to customer per message - communication breakdown needing human", ""),
    "277060_277059": ("order_status_delivery", False, "", "quoting tracking status verbatim, presumably with a question - ambiguous but delivery-topic clear"),
    "132381_132382": ("service_complaint_escalation", True, "explicit reference to having repeated the same info in previous response, unresolved", ""),
    "230002_230003": ("refund_or_return", True, "explicit demand for refund after 'so much trying' - repeated failed contact pattern", ""),
    "179118_179117": ("positive_feedback", False, "", "lighthearted positive exchange, no action needed"),
    "73538_73537": ("service_complaint_escalation", True, "no acknowledgment received, requesting faster response - communication failure needing human", ""),
    "142914_142912": ("payment_billing", True, "vague but concerning claim about bank account impact - needs human verification of what happened", ""),
    "132588_132586": ("other", False, "", "product availability question, not a support case"),
    "80521_80523": ("damaged_or_wrong_item", False, "", "package tampered/reopened, checked contents ok but flagging concern"),
    "277212_277210": ("order_status_delivery", True, "repeated failed contact - '4 times now' explicit count", ""),
    "206074_206072": ("order_status_delivery", False, "", "single missed delivery window, blames driver"),
    "179831_179829": ("payment_billing", True, "charged but item not received, and unable to modify order - financial dispute default escalate", "secondary: order_status_delivery"),
    "64056_64055": ("other", False, "", "asking for clarification on what a delivery guarantee term means - policy/definitional question"),
    "53494_53493": ("order_status_delivery", False, "", "prime late delivery disappointment, single mention"),
    "288071_288070": ("order_status_delivery", True, "hostile tone ('why should i suffer') + asking how long to wait, sustained frustration", ""),
    "130218_130216": ("other", True, "explicitly says standard app support doesn't apply, needs a specialized contact (regional publishing) - correctly falls outside taxonomy, route to human", ""),
    "150549_150548": ("order_status_delivery", False, "", "delivery date pushed later, single mention"),
    "63950_63951": ("other", False, "", "sarcastic short reply, insufficient standalone content"),
    "192298_192299": ("payment_billing", True, "unrecognized bank transaction claim - fraud-adjacent, policy default escalate", ""),
    "11059_11061": ("order_status_delivery", False, "", "brief follow-up on expected delivery time"),
    "22197_22195": ("order_status_delivery", True, "repeated failed contact - paid for Prime next-day, 5 days late, no callback yet", "secondary: prime_membership"),
    "169472_169471": ("payment_billing", False, "", "asking about payment/refund timing mechanics, informational tone"),
    "153620_153619": ("damaged_or_wrong_item", False, "", "misdelivery to wrong recipient, single mention, no explicit ask yet"),
    "199969_199968": ("other", True, "vague dissatisfaction with an unspecified 'investigation' - insufficient detail to classify or resolve without human context", ""),
    "19442_19440": ("positive_feedback", False, "", "brief thanks"),
    "36493_36492": ("digital_content_technical", False, "", "download failure, troubleshooting already attempted (restart/update/reinstall) - still not escalated since it's still within normal tech-support flow, but a good candidate for a 'troubleshooting already tried' rule (see report future work)"),
    "258791_258790": ("other", True, "accusation of dishonest review moderation practices - reputational/policy issue outside standard support scope, needs human", ""),
    "28427_28426": ("order_status_delivery", False, "", "very old preorder, can't view order status - unusual but still order-status-shaped"),
    "224784_224783": ("other", False, "", "product spec/generation question about Echo devices in a specific market"),
    "180932_180931": ("service_complaint_escalation", True, "explicit pattern claim ('this has happened') + questioning acceptability of service standard", ""),
    "258872_258874": ("positive_feedback", False, "", "received item, thanks"),
    "272670_272671": ("refund_or_return", False, "", "providing order ID for refund processing, explicitly declining DM link"),
    "248160_248159": ("order_status_delivery", True, "repeated pattern claim ('more than once') of broken next-day promise", "secondary: prime_membership"),
    "272895_272894": ("order_status_delivery", False, "", "asking about dispatch delay on a split shipment"),
    "288069_288067": ("service_complaint_escalation", True, "explicit high contact volume ('called so many times') with no resolution", ""),
    "14294_14292": ("service_complaint_escalation", True, "hostile generalized complaint about logistics quality ('laughably terrible')", ""),
    "205103_205102": ("service_complaint_escalation", True, "explicit claim that human reps have contradicted each other on a prior offered solution - needs supervisory review", ""),
    "142111_142110": ("other", False, "", "actually mixed-sentiment/sarcastic narrative post, hard to action directly, low stakes"),
    "73594_73596": ("refund_or_return", False, "", "asking for refund status update"),
    "183249_183248": ("other", False, "", "fragment referencing a linked screenshot, no standalone text content"),
    "247473_247474": ("service_complaint_escalation", True, "explicit frustration that repeated contact ('assurance to wait') isn't producing results", ""),
    "252822_252821": ("digital_content_technical", False, "", "device power issue, basic troubleshooting already partially tried"),
    "281162_281160": ("other", False, "", "product color/visual question referencing a link, low stakes"),
    "27841_27839": ("damaged_or_wrong_item", True, "high-value item left insecurely + safety/theft risk framing", ""),
    "69611_69612": ("cancel_order", False, "", "asking to cancel order to avoid return hassle, straightforward"),
    "280595_280597": ("order_status_delivery", True, "repeated failed contact - told to wait yesterday, still not delivered", ""),
    "257119_257121": ("service_complaint_escalation", True, "explicit repeated identical unhelpful reply pattern ('same reply yesterday and the day before')", ""),
    "86012_86014": ("refund_or_return", True, "explicit statement that troubleshooting was already done + demand for refund/compensation - past normal resolution path", ""),
    "276117_276115": ("other", False, "", "customer declining further engagement right now, no action needed"),
    "21045_21044": ("order_status_delivery", False, "", "single missed delivery promise, disappointed tone but first mention"),
    "77594_77592": ("service_complaint_escalation", True, "accusatory framing about brand's business practices and accountability", ""),
    "273876_273875": ("payment_billing", True, "AmazonPay cashback not received + explicit complaint about customer service response", "secondary: refund_or_return"),
    "253993_253992": ("other", False, "", "fragment stating an order was placed on a specific regional site, no explicit ask yet"),
    "204994_204993": ("service_complaint_escalation", True, "reports being redirected between brand and carrier over shipping cost responsibility - unresolved accountability dispute", ""),
    "158606_158605": ("other", True, "unrelated request to sell products via support channel + posts a phone number - route to human, off-scope and PII exposure risk", ""),
    "282451_282450": ("service_complaint_escalation", True, "accusation of dishonest business practice ('cheating people')", ""),
    "247004_247007": ("order_status_delivery", True, "explicit statement of still waiting for a reply to a prior message - repeated failed contact", ""),
    "222850_222849": ("other", True, "vague, no resolution received on an unspecified issue - insufficient context to classify without more chain history", ""),
    "262173_262174": ("digital_content_technical", False, "", "device UI/edit-option question"),
    "96688_96686": ("service_complaint_escalation", False, "", "reporting a competitor comparison unfavorably, venting rather than an active support request; kept non-escalate as no specific unresolved ask"),
    "95445_95444": ("digital_content_technical", False, "", "Alexa voice command behavior question"),
    "16853_16852": ("other", False, "", "fragment confirming something was already shared, insufficient standalone context"),
    "180473_180474": ("service_complaint_escalation", True, "hostile/insulting language toward support staff + explicit urgency demand", ""),
    "282485_293772": ("other", True, "references screenshots/messages not visible to us - insufficient context to classify or draft a grounded reply", ""),
}


def main():
    in_path = "data/golden/golden_sample_unlabeled.jsonl"
    out_path = "data/golden/golden_set.jsonl"
    rows = [json.loads(l) for l in open(in_path)]

    missing = [r["pair_id"] for r in rows if r["pair_id"] not in GOLD]
    if missing:
        print(f"WARNING: {len(missing)} pair_ids missing gold labels: {missing[:10]}")

    out = []
    for r in rows:
        gold = GOLD.get(r["pair_id"])
        if gold is None:
            continue
        intent, escalate, reason, notes = gold
        r["gold_intent"] = intent
        r["gold_should_escalate"] = escalate
        r["gold_escalate_reason"] = reason
        r["labeling_notes"] = notes
        out.append(r)

    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Labeled {len(out)}/{len(rows)} -> {out_path}")

    # summary stats
    from collections import Counter
    intent_counts = Counter(r["gold_intent"] for r in out)
    esc_counts = Counter(r["gold_should_escalate"] for r in out)
    print("\nIntent distribution:")
    for k, v in intent_counts.most_common():
        print(f"  {k:32s} {v:4d} ({v/len(out)*100:.1f}%)")
    print("\nEscalation distribution:", dict(esc_counts))


if __name__ == "__main__":
    main()
