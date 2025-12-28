# Legal Implementation Status (DSGVO/BGB)

Snapshot of current legal-related implementation and what’s still required.

## Imprint (Impressum)
- [ ] Phone number set in `lib/legal.ts` (`contact.phone`). Current placeholder: `+49 (0) 123 456789`. Must be replaced with the real, reachable number.
- [x] ODR link present (`https://ec.europa.eu/consumers/odr`) and VSBG note covered via translations.
- [ ] Responsible person name/email confirmed (check `legal.ts`/translations match reality).

## Widerruf (Right of Withdrawal)
- [x] Personalized goods exception (§312g Abs. 2 Nr. 1 BGB) present in translations.
- [ ] Visible pre-checkout hint/link that highlights the exception (e.g., above CTA in checkout flow).

## Cookie Consent
- [ ] Confirm `components/CookieConsent.tsx` blocks any non-essential scripts. If analytics or marketing are added, switch to a CMP (Usercentrics/Cookiebot) before enabling them.

## Checkout CTA Wording
- Current: “Unverbindlich vorbestellen” / “Submit pre-order (non-binding)” (translations). This is compliant for a pre-order/non-binding flow.
- [ ] If/when charging real payments, CTA must switch to “Kaufen” / “Zahlungspflichtig bestellen” (+ translations).

## Auftragsverarbeitung (AVV) Tracking
Maintain signed AVVs here (link to files or ticket IDs).

| Provider                 | Required | Status | Evidence/Link | Owner |
| ------------------------ | -------- | ------ | ------------- | ----- |
| Hetzner (Hosting)        | Yes      | ☐      |               |       |
| Google/Vertex/Gemini     | Yes      | ☐      |               |       |
| Stripe                   | Yes      | ☐      |               |       |
| Cloudflare (R2)          | Yes      | ☐      |               |       |

## Operational Reminders
- Run share-assets worker and gallery moderation so approved items stay compliant.
- Keep legal strings in `messages/*.json` aligned with `lib/legal.ts` data.
