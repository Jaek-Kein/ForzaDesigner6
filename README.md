# Forza Designer 6 (FD6)

<p align="center">
  <img src="tools/SplashScreen.gif" alt="Forza Designer 6 splash" width="600"/>
</p>

<p align="center">
  <img src="Pink.png" alt="FD6 badge" width="128"/>
</p>

> Convert any image into a Forza Horizon 6 vinyl group.

**Repo:** https://github.com/tokyubevoxelverse/ForzaDesigner6 · **License:** MIT · **Windows 10/11 x64**

---

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11, 64-bit |
| Game | Forza Horizon 6 |
| Disk | ~150 MB free |
| Microsoft Visual C++ Redistributable | Usually already installed. If FD6 fails to launch, get it [here](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

**Running from source** instead of the EXE additionally needs:
- Python 3.10+
- `pip install -r requirements.txt` (installs PySide6, NumPy, Pillow, PyInstaller, pytest)

---

## Install

1. Download `FD6.exe` from [Releases](https://github.com/tokyubevoxelverse/ForzaDesigner6/releases).
2. Double-click it. No installer, no admin rights.
3. Windows SmartScreen may warn the first time — click "More info" → "Run anyway".

---

## How to use

### A. Generate the shapes file

1. Launch `FD6.exe`.
2. Click **Upload Image…** and pick any JPEG/PNG.
3. In the **Settings** panel (right side):
   - Pick a **Profile** (`balanced` recommended).
   - Set **Stop at shapes** to your target (e.g. `1500` or `3000`).
4. Click **Start**. Watch the live preview rebuild your image.
5. When done, the `.json` is auto-saved next to your image. Click **Download JSON** if you want a copy elsewhere.

### B. Inject into Forza Horizon 6

1. Open **Forza Horizon 6** and open the **Vinyl Group editor**.
2. Load (or create) a template vinyl group with **at least N spheres**, where N = the shape count of your JSON. Tip: keep one or two reusable templates (1500-sphere and 3000-sphere) saved.
3. In FD6, click **Upload JSON** and pick your generated `.json`.
4. A modal dialog appears with live progress:
   - **Scanning FH6 memory** — 5–8 min first time per game session, seconds on later injections.
   - **Writing shapes** — ~10–30 sec.
5. When the status turns 🟢 green, close the dialog and check FH6 — your image is now the vinyl group.

---

## ✅ Do

- Open FH6 and the vinyl editor **before** clicking Inject.
- Keep at least one template vinyl group saved (sphere count ≥ your largest JSON).
- Let the injection finish — wait for the green status.
- Re-use the same JSON to re-inject if you mess up — generation only needs to happen once per image.
- Edit your vinyl group freely **between** injections.

## ❌ Don't

- **Don't edit, add, delete, or move any shape in FH6 during an active injection.** The dialog warns you. Editing reallocates the vinyl group's memory mid-write and the injection will fail. (Won't crash the game — just retry.)
- Don't close FH6 mid-injection.
- Don't run FD6 with admin rights "just in case" — it doesn't need them and may behave differently.
- Don't share your generated JSONs publicly without context — community etiquette: credit the source image.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Splash video hangs | Click anywhere or press Esc to skip. Hard 30-second auto-skip is always active. |
| "Patterns file not populated" warning on Inject | The `fh6_patterns.json` file was edited or corrupted. Restore it from the Git repo. |
| Inject finds 0 shape structs | FH6 patched and the vtable RVA shifted. See [Re-derive RVA](#re-derive-rva-after-an-fh6-update) below. |
| Inject says "template has X slots but JSON has Y" | Your template vinyl group doesn't have enough spheres. Load a bigger template. |
| Shapes appear offset / wrong scale | Known: FD6 currently writes raw pixel coords; FH6's canvas may use different units. Open an issue with a screenshot. |
| FH6 crashes after Inject | Should not happen — file an issue with your FH6 version and the JSON used. |

### Re-derive RVA after an FH6 update

The vtable RVA in `fd6/inject/patterns/fh6_patterns.json` is tied to a specific FH6 build. If a patch shifts it:

```powershell
# With FH6 open, vinyl editor active, one sphere placed at a known X coordinate:
python -m fd6.inject scan-float <known X>
python -m fd6.inject narrow <moved X>      # repeat until 1 hit
python -m fd6.inject dump <addr> 256
# vtable address = <addr> - 0x10  (struct start is 0x10 before the X field)
# new RVA = vtable_address - forzahorizon6.exe module base
# Edit anchor.vtable_rva in fh6_patterns.json with the new value (hex string, e.g. "0x6815CF8")
```

---

## Build from source

```powershell
git clone https://github.com/tokyubevoxelverse/ForzaDesigner6.git
cd ForzaDesigner6
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run from source:
python -m fd6

# Run tests:
pytest

# Build the single-file EXE:
.\build_exe.bat        # → dist\FD6.exe
```

---

## Credits

Inspired by the work of:
- **forza-painter** by `the_adawg` (FH4/FH5 — MIT)
- **geometrize-lib** by Sam Twidale (MIT)
- **Primitive** by Michael Fogleman (MIT)
- special thanks to bvzrays for his help with vinyl group layers detection and color offsets in allocated memory. helped a lot! https://github.com/bvzrays

FH6 vinyl-group memory layout was reverse-engineered from scratch for this project and is documented in [`fd6/inject/patterns/fh6_patterns.json`](fd6/inject/patterns/fh6_patterns.json) for community reuse.

## Disclaimer

**Use entirely at your own risk.** FD6 modifies the memory of a running Forza Horizon 6 process in order to populate vinyl-group shapes. While the tool does not patch the game executable, install drivers, modify save files, or attempt to bypass any anti-cheat or DRM system, **memory modification of a live game process may be interpreted by Microsoft, Xbox Live, or Turn 10 / Playground Games as a violation of the Microsoft Services Agreement, the Xbox Community Standards, or the Forza Horizon 6 terms of use. Doing so may result in temporary suspension or permanent ban of your Xbox / Microsoft account, loss of access to your purchased games, online services, achievements, and any content created with FD6.**

The author(s) and contributors of Forza Designer 6 **accept no responsibility or liability whatsoever** for any consequences arising from the use of this software, including but not limited to: account suspension or termination, loss of game progress, save corruption, loss of online services, hardware issues, or damages of any kind. By downloading, building, installing, or running FD6, you acknowledge that you understand these risks and accept them in full.

If you are uncertain whether using FD6 is acceptable to you, **do not run it.** This tool is provided as-is, MIT-licensed, with no warranties of any kind. Not affiliated with, endorsed by, or sponsored by Turn 10 Studios, Playground Games, Microsoft, Xbox, or any official Forza brand.

## License

[MIT](LICENSE) — free for any use with attribution.
