# Waggle Intent Codebook v1.2.0

> v1.2.0 changes (pilot round 2 residue): `\brma\b` (was firing inside
> `Normally`); same-sentence bound on `pay…issue`; `no update` →
> order-status; `after the update` → device-tech; media-question rule;
> `packagggg` elongation. Stratifier `heur-v1.2.0`. v1.1.0 history kept
> in git.
> v1.1.0 changes (from S2.2 pilot round 1, κ=0.397 → rewrite per gate):
> carrier/hub/pickup vocabulary for shipping; `haven't…got` + purchase-forms
> for order-status; typo tolerance (`compliant`, `appaling`, `packagggg`);
> profanity-as-vent markers; word-boundary on `available` (was firing inside
> `unavailable`); narrowed `how to` (was firing on `how do I post to
> Facebook`); non-English keyword sets (DE/FR/ES/PT) for shipping + refunds;
> account `get into my account` forms. Stratifier bumped to `heur-v1.1.0`.
> Irreducible residue: pure irony with zero vent words; Hindi-transliterated
> threads (stays `other` for the heuristic, human labels rule).

**Scope:** AmazonHelp customer-support Twitter threads (inbound customer turns only).
**Labels:** 10-way nominal. **Multi-intent rule:** log `multi_flag + second_best`,
adjudicate to the LOWEST priority index (priority 1 wins):

`account-security(1) > cancel-modify(2) > returns-refunds(3) > payment-promo(4) >
shipping-tracking(5) > order-status(6) > device-tech(7) > product-info(8) >
complaint-feedback(9) > other(10)`

**What counts as evidence:** the customer's own words. Agent boilerplate
(`DM us`, `cancel`, `courier`, sympathy templates) is NOT label evidence —
it appears in nearly every thread. Sarcasm (`Thanks amazon!` after a
complaint) does not change the underlying intent; log `hard` instead.
`ORD-12345`, `<URL>`, `@USER`, `__email__`, `__phone__` are atomic tokens.

**Blind protocol (S2.2):** label `pilot_50_blind.csv` from this codebook
only. Do not open `golden.jsonl` or any script output first. Disagreements
are adjudicated by the priority rule and logged in `adjudication_log.csv`.

---

## 1. account-security — login / access / fraud on the account itself

- **Definition:** cannot sign in, locked out, password/OTP/2FA problems,
  hacked or suspicious activity, phishing verification.
- **Include:** lockout, reset loops, unrecognized orders/charges as *account
  takeover* signals, "is this email legit?" phishing checks.
- **Exclude:** a late package on a healthy account → shipping/order-status;
  refund of a fraud charge where the ASK is money back → still
  account-security wins by priority (fraud domininates), note `second_best`.
- **Proto:**
  1. `have been locked out of my account without being notified. Can't access customer service as I need to login.`
  2. `second time somebody has hacked into my account and ordered a product. I've changed my password 3 times, help!`
  3. `my account has been locked for "security reasons" and there is no way to contact support?`
- **Edge:**
  1. `I just got an email that I ordered screen protectors, then changed the email on my account. I did none of these things.` → security (takeover), not cancel.
  2. `My account got hacked. Fraud charges not refunded. Customer service is the worst.` → security; `second_best=returns-refunds`.
  3. `Received an email from Amazon … to call … to authorize an order. Is this legit?` → security (phishing check), not order-status.

## 2. cancel-modify — stop or change an order/subscription before fulfilment

- **Definition:** explicit request to cancel, modify, edit, expedite or stop
  an order, subscription, Prime or shipment.
- **Include:** `cancel it/this`, `cancel … order within N words`, `expedite`,
  `modify/change/edit/update … order` (same sentence).
- **Exclude:** `app update` / `Touch ID update` (software, → device-tech);
  complaining ABOUT a cancellation Amazon already made → returns-refunds
  if the ask is redress, cancel-modify if the ask is reinstatement.
- **Proto:**
  1. `I have placed the order … under lighting deal & my order cancelled by amazon. Please deliver the cancelled items.`
  2. `Ordered his gift with 1 day delivery. Can I expedite it and receive it today?`
  3. `Cancelled last two orders … as it failed to reach me on time.`
- **Edge:**
  1. `the status of amazon orders are not being updated…` → order-status beats cancel (no cancel request; `second_best=shipping-tracking`).
  2. `The latest iOS app update breaks Touch ID for accessing orders.` → device-tech (`update` is software here).
  3. `Thanks amazon! [delayed item] didn't come with the controller skin.` → order-status + sarcasm; no cancel request.

## 3. returns-refunds — money or goods flowing back

- **Definition:** return, refund, RMA, chargeback, `money back`, send/take/ship
  `<item> back`, exchange, replacement-seeking, refund-status chasing.
- **Include:** `refund needed`, `charge … twice … give money straight back`,
  wrong-item returns (`received baby pants`).
- **Exclude:** `send feedback` (the word `feedback` contains `back` — NOT a
  return); fraud-charge refunds where takeover dominates → account-security
  with `second_best=returns-refunds`.
- **Proto:**
  1. `everytime I take out the issue of my return package, I am informed that Amazon received some 'baby pants' … i havent been issued a refu…`
  2. `Yes we did that still no refund` (refund chase).
  3. `Thank you. Refund done.` (confirmation of a completed refund).
- **Edge:**
  1. `I paid for Sunday delivery and then lied to about a delivery … Refund needed at the least!!` → refunds; `second_best=shipping-tracking`.
  2. `charge for me twice for 2 items yet dont give money straight back!` → refunds; `second_best=payment-promo`.
  3. `What’s the best method to send feedback regarding the physical Amazon Bookstore?` → product-info/other, NOT a return.

## 4. payment-promo — billing, charges, cashback, gift cards

- **Definition:** payment failures, double/unknown charges, invoices/bills,
  cashback/promo/coupon/gift-card problems.
- **Include:** `charged TWICE`, `no one can tell me what I am being billed for`,
  `Didn't get cashback`, missing gift card.
- **Exclude:** refund-status of an already-agreed refund → returns-refunds;
  fraud charges on a hijacked account → account-security.
- **Proto:**
  1. `i am doing payment on cash on delivery by visa card, everytime i see there is 10% cashback … i get nothing everytime why?`
  2. `failed to resolve my cashback issue. Cheated me with fake promotional offers`
  3. `How come no one can tell me what I am being billed for?`
- **Edge:**
  1. `My amazon account has been locked for two months because of fraudulent charges … I refuse to pay for.` → account-security wins; `second_best=complaint-feedback`.
  2. `I just got charged TWICE for a recent order... wtf. This is unacceptable.` → payment-promo; `second_best=shipping-tracking` only if delivery is also at issue.
  3. `Hi Amazon, I signed up … it says my email isn't registered?` → account-security/order-status, NOT billing.

## 5. shipping-tracking — where is it / when will it arrive

- **Definition:** tracking, carrier, shipped/in-transit/out-for-delivery
  states, late/delayed parcels, damaged-in-transit parcels where the ASK is
  delivery information.
- **Include:** `package hasn't shipped … arriving tomorrow`, `tracking shows
  delivered but nothing arrived`, `lasership`, `parcel … damaged … stickers`.
- **Exclude:** delivered-but-wrong-item redress → returns-refunds; pure
  status lookup with order number and no movement question → order-status.
- **Proto:**
  1. `My package hasn't shipped and it's scheduled to arrive tomorrow, is there any way to check if it'll still be delivered on time?`
  2. `my package has not arrived yet but ur tracking shows tat it shld have been delivered by now`
  3. `ORDER # ORD-12345 … prime eligible … Still after 4 days not shipped`
- **Edge:**
  1. `my pre order game … has still not been dispatched` → shipping; `second_best=order-status`.
  2. `got my delivery not in a box, damaged and stickers all over it?` → shipping (in-transit damage), not returns (no return asked).
  3. `choose slower shipping … delivers it early and it gets stolen` → shipping; theft narrative stays here unless a refund is demanded.

## 6. order-status — state of the order record (not the truck)

- **Definition:** order confirmation, pending/processing, missing
  confirmation email, `ORD-12345` lookups, `where/what … my order`,
  never-arrived orders framed as record problems.
- **Include:** `I ordered two products … and neither product arrived`,
  `not yet received order ORD-12345`, `what happened to my order?`
- **Exclude:** movement questions with tracking → shipping-tracking;
  cancellation requests → cancel-modify.
- **Proto:**
  1. `I ordered two products back at the end of July and neither product arrived.`
  2. `plz refer 2 my order ORD-12345. Status said additional info of address reqd.`
  3. `not yet received order ORD-12345 delivery date 17th Nov disgusting service`
- **Edge:**
  1. `any idea why lasership sucks? They didn't deliver a package …` → order-status; `second_best=complaint-feedback`.
  2. `worst service. didnt deliver the product but showing [shipped]` → order-status (record-vs-reality), not pure complaint.
  3. `Today is Oct 16...calling 8 day delivery 2 day shipping and charging for it is just a LIE` → shipping wins (movement promise); order-status second.

## 7. device-tech — Kindle / Echo / Alexa / Fire / app troubleshooting

- **Definition:** device setup, won't-turn-on, frozen, crash, reboot,
  wifi/bluetooth, firmware, app breaking a device-adjacent flow.
- **Include:** `Kindle breathed its last`, `Alexa … play a show`,
  `update is crashing every time I switch apps`, `change the music
  library country … Echo says configured wrong`.
- **Exclude:** `app update` that only changes order display with no device
  failure → order-status; warranty/availability questions → product-info.
- **Proto:**
  1. `looks like my Kindle breathed its last today. It just stopped working.`
  2. `How do I get Alexa to play a show on Netflix or Hulu?`
  3. `Your latest update is crashing every time I switch apps on my iPhone`
- **Edge:**
  1. `Why not an international warranty for Amazon Kindle?` → product-info; `second_best=device-tech`.
  2. `Can the UK control the Fire TV with Ama…` → device-tech question framed as capability → still device-tech; product-info second.
  3. `Comprei um e-book pelo aplicativo Kindle … e ele não aparece no meu Kindle` (Portuguese) → device-tech; flag non-English in notes.

## 8. product-info — pre-purchase questions, availability, compatibility

- **Definition:** what/which/compatible/size/stock/warranty/manual questions,
  `how to` instructions, availability (`nokia 8 … no cost emi`), listings.
- **Include:** `How to reschedule my order delivery?` (instruction-seeking),
  book listed on other stores but not here, delivery-window questions
  framed as policy (`delivery times of 7am-10pm`).
- **Exclude:** `How do I post this to Facebook` (not Amazon's product);
  `How do I get Alexa to…` (device operation → device-tech).
- **Proto:**
  1. `my book "Business Idea Blueprint" is listed on other Amazon stores but it's not available on <URL>`
  2. `please available nokia 8 on bajaj finsiv no cost emi`
  3. `How to reshedule my order delivery?`
- **Edge:**
  1. `How do I complain about a delivery that was "guaranteed"?` → complaint-feedback wins; product-info second (it asks how, but the act is a complaint).
  2. `thank you amazon for hiring exceptional employees lol` → other (sarcastic thanks, no product question).
  3. `Why do you give delivery times of 7am-10pm … Glad I'm a prime member... #joke` → product-info (policy question under sarcasm).

## 9. complaint-feedback — venting with no other actionable ask

- **Definition:** pure venting, insults, disappointment, `never again`,
  `pathetic`, `sucks` — where no refund/replacement/status/cancel is
  requested. This co-occurs with everything: only pick it when nothing
  above fits the ASK.
- **Include:** `getting a little pathetic`, `can someone call me … i have
  a complain`, `So disappointed … promised Oct 18-26 … now Oct 30` (no
  explicit ask → complaint, not shipping).
- **Exclude:** any concrete ask (refund → 3, redelivery → 5, cancel → 2).
- **Proto:**
  1. `2nd parcel that has "been delivered" but certainly not to me. getting a little pathetic.`
  2. `can someone from amazon call me … i have a complain … there is no result.`
  3. `Ordered #Sennheiser CX-180 … Hasn't even been 20 days and it has stopped working properly. Very disappointing`
- **Edge:**
  1. `I am usually a huge fan but was very disappointed today. Ordered … Advent calendar … came without [the Amazon box]` → complaint (packaging gripe, no ask).
  2. `what's the logic of giving fake hopes …` → complaint, not order-status (no lookup asked).
  3. `hang your head in shame. All of that packaging …` → complaint (packaging), even though `cancel your carbon footprint` contains the word cancel.

## 10. other — greetings, thanks, spam, unintelligible, off-topic

- **Definition:** everything else: greetings (`hello sir need help`),
  shout-outs (`S/O … for Sunday deliveries`), contests (`#AmazonPayAppQuiz
  who's the winner`), quiz/device chatter without a problem (`qiyi wuque
  … went to pieces` — a cube, not an Amazon product), non-English with no
  clear ask, empty/vague (`<URL>` only).
- **Include:** contest questions, praise, incomprehensible fragments.
- **Exclude:** if ANY actionable ask is present, prefer intents 1–9.
- **Proto:**
  1. `#AmazonPayAppQuiz who's the winner`
  2. `hello sir need help` (no problem stated)
  3. `S/O to @USER for Sunday deliveries`
- **Edge:**
  1. `attempted to deliver your parcel … REALLY? … #joke` → other is WRONG here if delivery is at issue — re-read: parcel complaint with sarcasm → shipping-tracking or complaint; log as edge showing sarcasm trap.
  2. `what happened to my order? #mixedsignals … 1 rep says on its way and 2 say it never was` → order-status (lookup asked), not other.
  3. `I just bout a qiyi wuque it went to pieces in 40 turns what the heck` → other (off-topic product).

---

## Edge bank (growing; append adjudication findings)

1. `feedback` contains `back` — never a return signal on its own.
2. `app update` is software (device-tech) unless the sentence also carries an order noun with a cancel/modify verb.
3. Agent boilerplate is not evidence: `DM`, `courier`, `disappointment`, `cancel` in agent turns must not flip the label.
4. Complaint co-occurs everywhere: only label complaint-feedback when no concrete ask exists.
5. Sarcasm (`Thanks amazon!`, `#joke`, `lol`) marks HARD, never changes the intent.
6. Non-English threads (PT/ES observed) keep their best-fit intent + note; pool is EN-majority post-S1.4.
7. `cancel it/this/them` with anaphoric reference counts as cancel-modify.
8. Short generics (`need help`, `hello`) with no problem stated are `other`, never account-security.
9. Typos are evidence: `compliant` = complaint, `appaling` = appalling, `packagggg` = package.
10. Bare `available` never matches inside `unavailable`; read the negation.
11. Carrier names (Hermes, DPD, AMZL, USPS…) + `hub` + `picked up` + `stuck`/`circulating` are shipping evidence.
12. `haven't/hasn't/never … got` and `purchase/bought/ordered … no update` are order-status evidence.
13. Non-English: DE (`Sendung/Packstation`), FR (`remboursé/colis`), ES/PT (`pedido/reembolso`) map to the same intents; note the language.
14. Pure irony with zero vent words (`Good proportional packaging…`) is complaint-feedback for humans, known heuristic miss — mark HARD.
15. `rma` is a word (`\brma\b`); `Normally` is not a return authorization.
16. `pay … issue` must be same-sentence: a far-away `issue` (agent's delivery company) does not make a billing case.
17. `no update / without update` and `after the update` (software broke something) are order-status / device-tech respectively.
18. Media questions (`where is the football game`, `how do people watch X`) are product-info (availability).
