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

## 5. Multi-Arch Infrastructure (Priority: Medium)
- [x] **Distributed Buildx Cluster**: Configured a 4-node cluster (AMD64/ARM64) for native multi-arch builds.
    - Nodes: `.16` (AMD64), `.12` (ARM64/Pi5), `.37` (AMD64), `.15` (ARM64/Jetson).
- [x] **Local Registry**: Set up an insecure local registry at `192.168.1.12:5000` for fast image distribution.
- [x] **Multi-Arch Build**: Successfully built and pushed `linux/amd64` and `linux/arm64` images to the local registry.
    - *Note*: Verified on Snapdragon X Elite (ARM64) and Pi5/Jetson.
- [ ] **Registry Security**: (Optional) Configure TLS for the local registry and update `buildkitd.toml` to remove `insecure = true`.
- [x] **Node Maintenance**: All 4 nodes are bootstrapped and active in the `distributed-builder`.
