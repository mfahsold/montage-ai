# Tactical Next Steps (Month 2: Shorts Studio 2.0)

See [ROADMAP_2026.md](./ROADMAP_2026.md) for full context.

## 1. Smart Reframing v2

- [x] **Subject Tracking**: Implement stable subject tracking (Kalman Filter + Path Planning).
- [x] **Motion Smoothing**: Improve `CameraMotionOptimizer` to reduce jitter in cropped vertical video.

## 2. Caption Styles

- [x] **Live Preview**: Enable instant preview of caption font/style changes in Web UI.
- [x] **Presets**: Hardcode "TikTok", "Cinema", and "Minimalist" CSS/FFmpeg distinct styles.

## 3. Highlights MVP

- [x] **Detection**: Combine Audio Energy + Visual Novelty to suggest "Top 5 Moments".
- [x] **Review Cards**: Add UI to let user accept/reject proposed highlights.
