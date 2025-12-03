# LUT Files for Color Grading

This directory contains 3D Look-Up Table (LUT) files for professional color grading.

## Supported Formats

- `.cube` - Adobe Cube LUT (most common)
- `.3dl` - Autodesk 3D LUT
- `.dat` - DaVinci Resolve LUT

## How to Use

1. **Download LUTs** from free sources:
   - [lutify.me](https://lutify.me) - Free cinematic LUTs
   - [iwltbap/Free-Luts](https://github.com/iwltbap/Free-Luts) - Open source collection
   - [Lutpack](https://www.lutpack.com) - Free packs
   - [Color Grading Central](https://www.colorgradingcentral.com/free-luts/) - Film emulations

2. **Place LUT files** in this directory

3. **Use in Creative Director** prompts:
   ```
   "Apply teal_orange_lut color grading"
   "Use film_emulation preset"
   ```

4. **Or via environment variable**:
   ```bash
   CREATIVE_PROMPT="cinematic look with film_emulation grading"
   ```

## Built-in Presets (No LUT Required)

These work out-of-the-box via FFmpeg filters:

| Preset          | Description                      |
| --------------- | -------------------------------- |
| `cinematic`     | Classic Hollywood look           |
| `teal_orange`   | Blockbuster complementary colors |
| `blockbuster`   | High contrast action movie       |
| `vintage`       | Faded film look                  |
| `film_fade`     | Lifted blacks, muted             |
| `70s`           | Warm retro                       |
| `polaroid`      | Instant camera style             |
| `cold`          | Blue/teal temperature            |
| `warm`          | Orange/yellow temperature        |
| `golden_hour`   | Sunset warmth                    |
| `blue_hour`     | Dawn/dusk cool                   |
| `noir`          | Black & white high contrast      |
| `horror`        | Desaturated, dark                |
| `sci_fi`        | Blue/cyan tech look              |
| `dreamy`        | Soft, ethereal                   |
| `vivid`         | Punchy, saturated                |
| `muted`         | Desaturated, filmic              |
| `high_contrast` | Strong blacks/whites             |
| `low_contrast`  | Flat, LOG-like                   |
| `punch`         | Saturated + sharp                |

## Expected LUT Filenames

If you add these specific files, they map to presets:

| Preset Name       | Expected File        |
| ----------------- | -------------------- |
| `cinematic_lut`   | `cinematic.cube`     |
| `teal_orange_lut` | `teal_orange.cube`   |
| `film_emulation`  | `kodak_2383.cube`    |
| `log_to_rec709`   | `log_to_rec709.cube` |
| `bleach_bypass`   | `bleach_bypass.cube` |

## Color Matching

When `COLOR_MATCH=true`, clips are automatically color-matched to the first clip in the sequence for visual consistency across the montage.
