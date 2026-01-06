# T14s Gen 6 (Snapdragon X Elite) Recovery + Ubuntu X1E Notes

## Scope and context
- Goal: install Windows 11 ARM on ThinkPad T14s Gen 6 (Type 21N1/21N2).
- Secondary goal: evaluate Ubuntu X1E (Snapdragon X Elite) with correct kernel/firmware.
- Repo hygiene: keep public repo clean (large downloads moved outside repo).

## Current status (Windows)
- Windows ARM installer boots, but setup fails with "No drivers were found".
- In WinPE, the USB stick does not appear, which suggests missing USB controller/storage
  drivers in the WinPE environment (not just missing NVMe).

## Actions taken so far (Windows)
1) Ventoy multi-boot
   - Normal mode froze after selection. Ventoy docs note Normal mode relies on
     ISO emulation and can fail on some firmware; WIMBOOT is recommended for
     official Windows ISOs.
   - References:
     - https://www.ventoy.net/en/doc_wimboot.html
     - https://www.ventoy.net/en/doc_grub2boot.html
     - https://github.com/ventoy/Ventoy/issues/2688

2) Dedicated Windows ARM USB (no Ventoy)
   - Wiped and repartitioned USB as GPT + FAT32 (single partition, label WINARM).
   - Copied Windows ARM ISO contents; split install.wim into install.swm parts
     (FAT32 4GB file limit).
   - Added Lenovo driver pack and Qualcomm firmware on the USB for manual loading.

3) Injected drivers into WinPE (boot.wim)
   - Added Lenovo SCCM driver pack to boot.wim (indexes 1 and 2) and
     auto-loaded drivers at WinPE startup via startnet.cmd.
   - Still did not surface USB in WinPE, implying missing WinPE-specific drivers.

4) Lenovo WinPE-specific driver pack
   - Lenovo provides a dedicated WinPE driver pack for this model:
     "SCCM Package for Windows 11 PE - ThinkPad T14s Gen 6 (21N1/21N2)".
   - Downloaded and injected that pack into boot.wim (indexes 1 and 2).
   - Also copied it to USB at Drivers/lenovo/winpe for manual drvload.
   - Source:
     - https://download.lenovo.com/pccbbs/mobiles/tp_8cxg4_pe11_202507.exe
     - https://download.lenovo.com/pccbbs/mobiles/tp_8cxg4_pe11_202507.html

## Hypothesis (Windows)
- The WinPE environment is missing Qualcomm USB controller drivers or
  storage stack components required to enumerate the USB or NVMe.
- Lenovo OEM recovery media (DDRS) should include all required WinPE drivers.

## Lenovo Recovery USB (DDRS) path
- Lenovo Digital Download Recovery Service (DDRS) can generate OEM recovery
  media with the correct WinPE drivers baked in.
- Requires Windows tools (Digital Download Tool + USB Recovery Creator).
- Key details from Lenovo DDRS docs:
  - https://download.lenovo.com/pccbbs/thinkcentre_pdf/lenovo_digital_download_recovery_service-ddrs.pdf
  - https://support.lenovo.com/lenovorecovery/entitlement

## Local archive location (outside repo)
- /home/codeai/boot_archives/boot_media_20260104_143614
  (contains ISO files, Lenovo driver packs, and previous USB artifacts)

## Ubuntu X1E research (Snapdragon X Elite)
### Canonical "Ubuntu Concept" images
- Canonical publishes experimental Ubuntu concept images for Snapdragon X Elite.
- The announcement notes this is a developer preview and experimental; the
  image is known to work best on Lenovo ThinkPad T14s Gen 6.
- ISO links (always updated):
  - https://people.canonical.com/~platform/images/ubuntu-concept/plucky-desktop-arm64+x1e.iso
  - https://people.canonical.com/~platform/images/ubuntu-concept/oracular-desktop-arm64+x1e.iso
- Source: https://discourse.ubuntu.com/t/ubuntu-concept-snapdragon-x-elite/48800

### Ubuntu 25.04 generic arm64 ISO (official)
- Ubuntu 25.04 generic arm64 ISO now works on many Snapdragon X Elite devices,
  including ThinkPad T14s Gen 6, but hardware support is still WIP.
- Recommended: update firmware first, keep Windows for firmware updates,
  dual-boot, and check "include additional drivers" in installer.
- Firmware files may be required for some devices; for T14s Gen 6 this should
  work out of the box.
- For devices with >32GB RAM, the concept ISO includes a workaround that is
  not in the official installer.
- Source: https://discourse.ubuntu.com/t/faq-ubuntu-25-04-on-snapdragon-x-elite/61016

### Kernel and driver track (X1E)
- Ubuntu Concept uses a qcom-x1e kernel (linux-qcom-x1e) and a dedicated PPA.
- How to build the kernel from Ubuntu Concept sources:
  - https://discourse.ubuntu.com/t/how-to-build-your-own-ubuntu-concept-x-elite-kernel/49941
- PPA: https://launchpad.net/~ubuntu-concept/+archive/ubuntu/x1e

### Firmware extraction (if needed)
- If device-specific firmware is missing, Ubuntu recommends:
  - sudo apt install qcom-firmware-extract
  - sudo qcom-firmware-extract
- Source: https://discourse.ubuntu.com/t/faq-ubuntu-25-04-on-snapdragon-x-elite/61016

## Proposed Ubuntu X1E test plan (T14s Gen 6)
1) Start with Ubuntu Concept 25.04 X1E ISO (plucky) to maximize hardware support.
2) Update BIOS/firmware first in Windows, then boot the ISO.
3) Use dual-boot and keep Windows for future firmware updates.
4) In the installer, check "include additional drivers" and connect to network.
5) After boot:
   - Verify ubuntu-drivers list returns hwe-qcom-x1e-meta.
   - If missing, check /sys/devices/virtual/dmi/id/modalias and report via
     Ubuntu Concept tracking bugs.
6) Only if required: try the official Ubuntu 25.04 arm64 ISO and compare.

## Open questions / next decisions
- If WinPE still cannot see USB after the Lenovo WinPE pack injection,
  proceed with Lenovo DDRS recovery media (requires Windows environment).
- Decide whether to test Ubuntu on the concept ISO first or directly on
  official 25.04 arm64 (concept recommended for best support and >32GB RAM).

## Sources (quick list)
- Ventoy WIMBOOT mode: https://www.ventoy.net/en/doc_wimboot.html
- Ventoy GRUB2 mode: https://www.ventoy.net/en/doc_grub2boot.html
- Ventoy Lenovo board issue (Normal mode): https://github.com/ventoy/Ventoy/issues/2688
- Lenovo DDRS PDF: https://download.lenovo.com/pccbbs/thinkcentre_pdf/lenovo_digital_download_recovery_service-ddrs.pdf
- Lenovo DDRS entitlement: https://support.lenovo.com/lenovorecovery/entitlement
- Ubuntu Concept X Elite announcement: https://discourse.ubuntu.com/t/ubuntu-concept-snapdragon-x-elite/48800
- Ubuntu 25.04 X Elite FAQ: https://discourse.ubuntu.com/t/faq-ubuntu-25-04-on-snapdragon-x-elite/61016
- Ubuntu Concept kernel build: https://discourse.ubuntu.com/t/how-to-build-your-own-ubuntu-concept-x-elite-kernel/49941
