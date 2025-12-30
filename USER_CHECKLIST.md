# User Checklist (Manual Tasks)

This list contains tasks that require manual intervention, access to external systems, or sensitive operations that cannot be automated by the AI agent.

## 1. Security & Repo Hygiene (Priority: High)
- [ ] **Purge Git History**: Run `git filter-branch` or `BFG Repo-Cleaner` to completely remove `docs/marketing/monetization_strategy.md`, `docs/marketing/social_media_strategy.md`, and `docs/marketing/positioning.md` from the entire commit history.
    - *Why*: Simply adding them to `.gitignore` does not remove them from previous commits.
- [ ] **Rotate Secrets**: If any API keys were ever committed (even in now-deleted files), rotate them immediately.

## 2. Cloud Infrastructure Setup (Priority: Medium)
- [ ] **Set up Billing Provider**: Choose and configure a payment provider (Stripe/Paddle) for the "Pro" tier.
- [ ] **Deploy Cloud Backend**: Set up the actual `api.montage.ai` endpoint that `CloudConfig` points to.
- [ ] **Generate API Keys**: Create a mechanism to issue `MONTAGE_CLOUD_API_KEY`s to paying users.

## 3. Legal & Compliance (Priority: Medium)
- [ ] **Finalize Terms of Service**: Update the website with a proper ToS that covers the "Source Available" license and Cloud usage.
- [ ] **Privacy Policy**: Update to include data handling for Cloud GPU offloading (what data is sent, how long it's stored).

## 4. Marketing & Launch (Priority: Low - 1 Month out)
- [ ] **Restore Strategy Docs**: Once the repo history is clean, restore the strategy documents into the `private/` folder (locally) for your own reference.
- [ ] **Prepare Launch Content**: Create the "Before vs. After" videos mentioned in the social media strategy.
