# Artist ELO Ranking System for NovelAI

A web-based blind comparison system that ranks Danbooru artist tags by generating AI images with [NovelAI](https://novelai.net/) and letting you pick your preferred results. Artists gain or lose ELO rating based on the outcomes.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Blind Comparisons**: Compare two AI-generated images without seeing which artists were used (toggleable)
- **ELO Rating System**: Individual-based ELO calculation with zero-sum enforcement
- **Active Pool Management**: Moves between roughly 150-200 artists for efficient discovery and replacement
- **Ranking Views**: Switch main-pool sampling between Default, Top-tier Battles, and Bottom-tier Battles
- **Smart Rotation**: Low performers are probabilistically rotated out; high ELO artists are more likely to return
- **Two Independent Pools**: Keep a main-pool ELO and a separate Hall of Fame ELO
- **Hall of Fame Controls**: Use image-level star controls to promote or restore complete artist combinations
- **Single Deathmatch Queue**: Broken-heart combinations move to solo UP/DOWN review instead of being removed immediately
- **Hall of Fame Modes**: Compare Hall of Fame artists as solo tags or normal 1-3 artist combinations
- **Weighted Combination Lab**: Compare disjoint 3-10 artist Hall of Fame teams with NovelAI weights from 0.5 to 2.0
- **Undo Support**: Revert the last comparison if you change your mind
- **Custom Prompts**: Use your own positive and negative prompts
- **NovelAI-Style Settings**: Resolution, steps, guidance, seed, sampler, Variety+, guidance rescale, noise schedule, quality tags, and UC presets
- **Fair Pair Seeds**: Both images use the same seed in each comparison to reduce luck-based differences
- **Prompt Presets**: Save and reload the prompt, negative prompt, and every image setting in 10 persistent slots
- **Temporary Artist Intake**: Extract known artist tags from pasted text, force one into each normal combination, and automatically add evaluated artists to the main pool
- **Anlas Balance**: Show the current NovelAI Anlas balance without external purchase links
- **Export to CSV**: Download full leaderboard with detailed stats
- **Comparison History**: View your last 10 comparison results

## Prerequisites

- Python 3.8 or higher
- A [NovelAI](https://novelai.net/) subscription with API access
- The artist tags file (`danbooru_artist_tags_v4.5.txt`)

## Personal Mobile Web

The interface includes a mobile-first layout, large touch voting controls, optional login protection, and persistent data-directory support. A Dockerfile and Render Blueprint are included for running it without keeping a PC online.

See **[MOBILE_DEPLOY.md](MOBILE_DEPLOY.md)** for the Korean phone-only deployment guide. Never commit your NovelAI token or app password to GitHub.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rlag1998/novelai-artist-elo.git
   cd novelai-artist-elo
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**

   Copy the example environment file and add your NovelAI API key:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and replace `your-api-key-here` with your actual API key.

## Getting Your NovelAI API Key

1. Go to [NovelAI](https://novelai.net/) and log in
2. Navigate to **Account Settings**
3. Click **Get Persistent API Token**
4. Copy the token (starts with `pst-`)
5. Paste it into your `.env` file

> **Note**: API access requires an active NovelAI subscription (Tablet, Scroll, or Opus tier).

## Usage

1. **Start the application**
   ```bash
   python artist_elo_ranker.py
   ```

2. **Open the web interface**

   The application will automatically open your browser to `http://127.0.0.1:7860`

3. **Compare images**

   Two images will be generated with different artist combinations (1-3 artists each). Pick your preferred image to update the ELO ratings.

   ![Comparison View](screenshots/picker.png)

   - Tap **A 선택** or **B 선택** to vote for your preferred image
   - Use **건너뛰기 / 새 비교** to generate a new pair without changing ELO
   - Use **마지막 선택 되돌리기** to restore the prior vote and pair
   - Switch between the **랭킹** and **가중치 조합** tabs without losing either page's current images
   - In the main pool, choose **기본**, **상위권 대결**, or **하위권 대결** before generating the next pair

4. **View rankings**

   The ranking panel shows the selected pool's own ELO and comparison counts. Main-pool and Hall of Fame ratings never overwrite each other.

   <img src="screenshots/leaderboard.png" width="350">

### Understanding the Leaderboard

Each artist entry shows:
```
1. artist_name 1650 · 25회
```

- **1650** - Current ELO rating
- **25회** - Comparisons recorded in the currently selected ELO pool

### Two Pools and Image Actions

Choose **전체 작가 풀**, **명예의 전당**, or **단일 데스매치** above the ranking images.

- In the main pool, `☆` moves every artist used by that image into the Hall of Fame. Their Hall of Fame ELO and comparison count start again at 1500 and zero, while the existing main ELO is preserved.
- A filled `★` returns those artists to the main pool and restores the preserved main ELO. Entering the Hall of Fame again always starts a new 1500-point Hall of Fame run.
- The empty broken-heart control sends every artist used by that main-pool image to the persistent single-artist deathmatch queue. It does not erase their main ELO.
- The deathmatch screen generates one solo image for each A/B candidate. Judge each side independently: **UP** restores that artist to the main pool with the existing ELO, while **DOWN** confirms pool-out without deleting the stored ELO. An odd final candidate is shown by itself.
- Pool actions resolve the current image pair without generating another image. Use the new-comparison button when ready, avoiding an unintended NovelAI charge.

### Hall of Fame and Weighted Modes

The first page offers two Hall of Fame comparison modes:

- **단일 작가**: one Hall of Fame artist per image
- **일반 조합**: one to three Hall of Fame artists per image

The second page builds two non-overlapping Hall of Fame teams. Each side contains
the selected 3-10 artists and uses the same total prompt weight, with individual
weights sampled from 0.5 to 2.0. Generated tags use NovelAI syntax:

```text
1.3::artist: alpha::, 0.8::artist: beta::, 1.1::artist: gamma::
```

Hall of Fame ELO affects which artists are grouped together. The effective team
rating is the prompt-weighted average ELO, and the result gives a bounded larger
share of credit or blame to higher-weight artists while keeping the total ELO
change zero-sum. A 3-vs-3 comparison needs at least six Hall of Fame artists;
10-vs-10 needs twenty.

### Temporary Artist Discovery

Open **Temporary Artist Discovery**, then paste a comma- or newline-delimited
artist list, prompt, or a copied `Name / Cosine / Jaccard / Overlap / Frequency`
statistics table. For table rows, the app automatically removes the leading `?`,
post-count values such as `533` or `1.0k`, and all metric columns. It then keeps
only exact matches from the bundled artist tag file, accepts `artist: name`,
prompt-weight wrappers, and space/underscore variants, removes duplicates, and
discards every unregistered or non-artist segment.

After at least two artists are recognized, start temporary intake. Every following
main-pool A/B combination contains at least one distinct temporary artist, while
the remaining positions may contain normal main-pool artists. Combination sizes
remain 1-3, so a natural solo result is still possible.

After a vote, the temporary artists used on both sides are automatically added to
the main pool and removed from the temporary queue. Their first main ELO result is
that vote; an already-rated artist keeps its stored main ELO. Undo restores both
the rating and temporary queue. Stopping intake keeps the cleaned list for later.
The list and enabled state are stored in `temporary_pool.json`.

### Pool Health & Statistics

The badge above each image pair shows the main-pool count, Hall of Fame count,
deathmatch queue count, confirmed pool-out count, and remaining temporary intake count.
The ranking panel reports the selected pool's comparison total and top 30 members.

### Export to CSV

Click "Export Leaderboard as CSV" to download the full leaderboard as a CSV file with detailed statistics:

<img src="screenshots/export_leaderboard.png" width="400">

- Main rank, artist, and current membership (`main`, `hall_of_fame`, `deathmatch`, or `out`)
- Main ELO/comparisons and separate Hall of Fame ELO/comparisons
- Wins, losses, and win rate
- Solo, normal, and weighted-mode appearance counts
- Average prompt weight used in weighted Hall of Fame comparisons

The export includes ALL rated artists (not just the top 30 shown in the UI).

### Comparison History

The "Comparison History" accordion shows your last 10 comparison results (newest first), displaying which artist combinations won against which. Useful for reviewing your choices and spotting patterns.

<img src="screenshots/comparison_history.png" width="400">

### Custom Prompts & Generation Settings

Open the **Image Settings** accordion to access the NovelAI-style generation panel.

![Custom Prompt Editor](screenshots/custom_prompt.png)

#### Positive Prompt
- Write any NovelAI prompt you want to use for comparisons
- The system will automatically insert artist tags into your prompt
- If your prompt contains `{artist_placeholder}`, artist tags replace that marker
- If no placeholder exists, artist tags are appended to the end
- Leave empty to use the built-in default prompt

**Example:**
```
1boy, fantasy warrior, armor, castle background, {artist_placeholder}, masterpiece
```

#### Negative Prompt
- Enter your own negative prompt to control what to avoid in generations
- Leave empty to use the built-in default negative prompt
- Artist tags are NOT inserted into negative prompts

#### Add Quality Tags (Checkbox)
When enabled (default), automatically appends to your positive prompt:
```
very aesthetic, masterpiece, no text
```

#### Auto-Negative Preset (Dropdown)
NovelAI can automatically add preset tags to your negative prompt. Choose from:

| Preset | Description |
|--------|-------------|
| **None** | Disabled - no auto-negative tags added |
| **Heavy** | Standard quality filters (lowres, bad quality, jpeg artifacts, etc.) |
| **Light** | Minimal filters (shorter list, less restrictive) |
| **Human Focus** | Optimized for character art (sketch, flat colors, comic, etc.) |
| **Heavy + Anatomy** | Heavy preset plus body/anatomy fixes (bad anatomy, glowing eyes, etc.) |

**Tips:**
- Keep prompts consistent during a ranking session for fair comparisons
- The same prompt, generation settings, and seed are used for both images (only the artists differ)
- Test how artists perform with specific subjects (landscapes vs portraits)

#### Image Settings

- Resolution presets: Normal Square, Normal Portrait, and Normal Landscape
- Steps: 1-50
- Prompt Guidance: 0-10
- Seed: leave blank for a new random seed per comparison, or enter a number to lock it
- Sampler: Euler Ancestral, DPM++ 2M, or Euler
- Variety+, Prompt Guidance Rescale, and Noise Schedule
- Each side always generates exactly one image for an unambiguous A/B vote

#### Prompt Presets

Choose slot 1-10, then tap **💾** to save or **📂** to load. A slot stores the
positive prompt, negative prompt, resolution, steps, guidance, seed, sampler,
Variety+, guidance rescale, noise schedule, quality toggle, and UC preset. Presets
are stored in `prompt_presets.json` under `DATA_DIR` and survive redeploys when a
persistent disk is mounted.

## Artist Tags File

The system requires a text file containing Danbooru artist tags, one per line. The default filename is `danbooru_artist_tags_v4.5.txt`.

Example format:
```
asanagi
wlop
ilya_kuvshinov
sakimichan
```

You can create your own list or find Danbooru artist tag compilations online.

## Configuration

All configuration is done via environment variables in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `NOVELAI_API_KEY` | (required) | Your NovelAI API token |
| `NAI_STEPS` | 28 | Number of diffusion steps |
| `NAI_IMG_WIDTH` | 1024 | Image width in pixels |
| `NAI_IMG_HEIGHT` | 1024 | Image height in pixels |
| `NAI_PROMPT_GUIDANCE` | 5.0 | Default prompt guidance |
| `NAI_GUIDANCE_RESCALE` | 0.0 | Default prompt guidance rescale |
| `NAI_SAMPLER` | k_euler_ancestral | Default sampler key |
| `NAI_NOISE_SCHEDULE` | karras | Default noise schedule |
| `ELO_DEFAULT` | 1500 | Starting ELO for new artists |
| `ELO_K_FACTOR` | 32 | ELO K-factor (rating volatility) |
| `POOL_SIZE` | 150 | Active pool size |
| `SERVER_HOST` | 127.0.0.1 | Server bind address |
| `SERVER_PORT` | 7860 | Server port |
| `APP_USERNAME` | artist | Personal web login username |
| `APP_PASSWORD` | (optional locally) | Required, minimum 8 characters when exposed online |
| `DATA_DIR` | Project directory | Persistent directory for rankings, history, and images |
| `INBROWSER` | true | Automatically open a local browser on startup |
| `DEFAULT_PROMPT` | (built-in) | Default positive prompt. Use `{artist_placeholder}` for artist tags. Wrap in quotes. |
| `NEGATIVE_PROMPT` | (built-in) | Default negative prompt. Wrap in quotes. |

## How ELO Ranking Works

The system uses an individual-based ELO calculation:

1. **Team Formation**: Each image uses 1-3 randomly selected artists from the active pool
2. **Comparison**: You pick your preferred image in a blind comparison
3. **ELO Update**: Each winning artist gains ELO based on their individual rating vs. the losing team's average
4. **Zero-Sum**: Total ELO gained equals total ELO lost (scaled for fairness)
5. **Pool Rotation**: Underperformers may be rotated out; high-ELO artists are more likely to return

Hall of Fame comparisons use a different ELO file. Weighted comparisons replace
the opposing team's simple average with a prompt-weighted average and scale each
artist's bounded influence by its 0.5-2.0 prompt weight.

**Note:** If the same artist appears on both sides of a comparison, they are excluded from ELO changes (they can't win or lose against themselves). Skipping a comparison also results in no ELO changes.

### Pool Rotation Strategy

- **Removal Weight**: `confidence * underperformance²`
  - Confidence: matches / 5 (capped at 1.0)
  - Underperformance: relative to pool's best performer
- **Addition Weight**: `(ELO - min_ELO + 100)²`
  - Squared preference for high-ELO artists
- **Explicit exclusions**: Broken-heart artists and current Hall of Fame members are not eligible for automatic main-pool refill

## Data Files

The application creates/uses several JSON files:

| File | Purpose |
|------|---------|
| `artist_elo_ratings.json` | Main-pool ELO ratings and comparison counts |
| `active_pool.json` | Main-pool membership and explicit broken-heart exclusions |
| `hall_of_fame_pool.json` | Current Hall of Fame membership |
| `hall_of_fame_elo_ratings.json` | Independent Hall of Fame ELO, mode counts, and weighted exposure |
| `temporary_pool.json` | Pasted artist intake queue and enabled state |
| `prompt_presets.json` | Ten prompt and image-setting preset slots |
| `current_comparison.json` | Current images, actual pair seed, and generation settings |
| `weighted_comparison.json` | Independent current weighted-page images, artists, and prompt weights |
| `comparison_history.json` | Full history of all comparisons |

These files are automatically created on first run and persist your rankings across sessions.

## Project Structure

```
novelai-artist-elo/
├── artist_elo_ranker.py      # Main application
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment file
├── .env                     # Your environment file (create this)
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT license
├── README.md                # This file
├── danbooru_artist_tags_v4.5.txt  # Artist tags (you provide)
├── comparison_images/       # Generated images (auto-created)
├── artist_elo_ratings.json  # ELO data (auto-created)
├── active_pool.json         # Pool data (auto-created)
├── hall_of_fame_pool.json   # Hall of Fame membership (auto-created)
├── hall_of_fame_elo_ratings.json # Hall of Fame ELO (auto-created)
├── temporary_pool.json      # Temporary intake queue (auto-created)
├── weighted_comparison.json # Weighted-page state (auto-created)
└── comparison_history.json  # History (auto-created)
```

## Tips

- **Start with more comparisons**: New artists need ~5 comparisons before they can be confidently rotated out
- **Use consistent prompts**: Changing prompts mid-session can affect rating fairness
- **Build the Hall gradually**: Weighted mode needs at least six Hall of Fame artists
- **Keep team sizes equal**: The weighted page automatically uses equal artist counts and total prompt weight on both sides

## Troubleshooting

### "NOVELAI_API_KEY environment variable is not set"
- Make sure you created a `.env` file with your API key
- Verify the key is on a line starting with `NOVELAI_API_KEY=`

### "Artist tags file not found"
- Place your `danbooru_artist_tags_v4.5.txt` file in the same directory as the script
- Or modify `ARTIST_TAGS_FILE` in `config.py` to point to your file

### Images fail to generate
- Check your NovelAI subscription is active
- Verify your API key is correct
- Check your internet connection

### Windows: Compilation errors during pip install
If you see errors like `Unknown compiler(s): [['cl'], ['gcc'], ['clang']...]` when installing dependencies, you're missing C++ build tools.

**Fix:**
1. Download [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer and select **"Desktop development with C++"** workload
3. Install and restart your terminal
4. Try `pip install -r requirements.txt` again

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [NovelAI](https://novelai.net/) for the image generation API
- [Gradio](https://gradio.app/) for the web interface framework
- [Danbooru](https://danbooru.donmai.us/) for artist tag conventions
