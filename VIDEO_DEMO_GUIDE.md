# Video Demo Creation Guide

**Purpose:** Create 3-5 minute video demonstration for HACK4EARTH BUIDL Challenge
**Target:** Maximize Openness & Transparency category score (17/20 → 20/20)
**BONUS Points:** +3 points for video demo

---

## 🎬 Video Structure (3-5 minutes)

### 1. Introduction (30 seconds)
**Script:**
```
"Hi, I'm Ramesh Arvind, and this is EcoGrow - a physics-informed AI system 
that achieves 76.5% energy reduction in greenhouse control.

We're addressing HACK4EARTH's dual challenge: making AI more sustainable 
through quantization (Track A), while using AI to advance agricultural 
sustainability through carbon-aware scheduling (Track B)."
```

**Screen:** 
- Show GitHub repository homepage with badges
- Quick pan through README.md sections

---

### 2. Problem Statement (30 seconds)
**Script:**
```
"Traditional greenhouse AI models consume massive energy - 0.162 kWh per 
1000 inferences. Scaled to 1000 greenhouses, this generates 24,640 tons 
of CO₂ annually - equivalent to 5,356 cars on the road.

Our solution: Dynamic INT8 quantization + carbon-aware scheduling."
```

**Screen:**
- Show baseline_benchmark.py results
- Display evidence.csv with highlighted baseline energy values

---

### 3. Live Demo - QuickStart Notebook (90 seconds)
**Script:**
```
"Let me show you our hardware-validated results. I'm running the QuickStart 
notebook which calculates Track A and B metrics from real measurements."
```

**Actions:**
1. Open `notebooks/01_QuickStart_Submission.ipynb`
2. Run Cell 1 (imports) - narrate: "Loading evidence data from 22 hardware runs"
3. Run Cell 2 (load data) - show evidence_df table
4. Run Cell 3 (Track A calculations) - highlight:
   - Baseline: 0.162 kWh
   - Optimized: 0.038 kWh
   - **76.5% reduction**
5. Run Cell 4 (Track B calculations) - highlight:
   - Carbon-aware savings: 43.2%
   - Cost reduction: 49.3%
6. Run Cell 5 (submission export) - show submission.csv created

**Screen:**
- Zoom into key cells showing calculations
- Pause on visualizations (if present)
- Show CSV file opened in text editor

---

### 4. Carbon-Aware Scheduling Demo (60 seconds)
**Script:**
```
"Our carbon-aware scheduler shifts AI workloads to solar peak hours - 
10am to 4pm - when grid carbon intensity drops from 420 to 160 grams 
per kilowatt-hour. That's an additional 43% carbon reduction."
```

**Actions:**
1. Open `notebooks/02_CarbonAware_Demo.ipynb`
2. Run cells showing:
   - 24-hour carbon intensity profile (ENTSO-E data)
   - Optimal scheduling window (green highlight)
   - Peak hours to avoid (red highlight)
3. Show decision table with savings per task

**Screen:**
- Focus on carbon intensity visualization
- Highlight decision logic code
- Show JSON output with task scheduling

---

### 5. Scaling Impact (45 seconds)
**Script:**
```
"Now let's see the real-world impact. At scale - 1000 greenhouses, 10,000 
square meters each - we're saving:
- 24,640 tons of CO₂ annually
- 2 million cubic meters of water
- Protecting 502,500 people through food security

With a payback period of just 4 months."
```

**Actions:**
1. Open `notebooks/04_Impact_Analysis_Extended.ipynb`
2. Run cells showing:
   - Water impact calculations
   - People protected metrics
   - Multi-metric dashboard visualization
3. Show impact_math_extended.csv

**Screen:**
- Show dashboard with all 4 metrics
- Highlight economic ROI (0.32 years)
- Pan through CSV showing water/people columns

---

### 6. Live Dashboard (30 seconds)
**Script:**
```
"We've also built a real-time monitoring dashboard using Streamlit. 
Here you can see live energy consumption, carbon intensity profiles, 
and per-component breakdowns."
```

**Actions:**
1. Run `./launch_dashboard.sh` in terminal
2. Show dashboard at http://localhost:8501
3. Switch between views:
   - Real-Time Monitoring
   - Historical Analysis
   - Impact Scenarios

**Screen:**
- Show gauges updating
- Switch scenario dropdown
- Highlight 24-hour carbon profile

---

### 7. Conclusion & Call to Action (15 seconds)
**Script:**
```
"EcoGrow proves that sustainable AI is not just possible - it's profitable. 
All code, data, and measurements are open-source on GitHub. 

Let's make AI green together. Thank you!"
```

**Screen:**
- Show GitHub repository with star button
- Display README.md bottom section with badges
- End with project logo/title card

---

## 🛠️ Technical Setup

### Required Tools
1. **Screen Recording:**
   - **Linux:** SimpleScreenRecorder, OBS Studio, Kazam
   - **Windows:** OBS Studio, Camtasia
   - **macOS:** QuickTime Player, ScreenFlow

2. **Video Editing:**
   - **Free:** OpenShot, Shotcut, DaVinci Resolve
   - **Paid:** Adobe Premiere Pro, Final Cut Pro

3. **Audio:**
   - Use external microphone for clear narration
   - Background music (optional): royalty-free from YouTube Audio Library

### Recording Settings
```
Resolution: 1920x1080 (Full HD)
Frame Rate: 30 fps
Format: MP4 (H.264)
Audio: AAC, 128 kbps
Bitrate: 5-8 Mbps
```

---

## 📋 Pre-Recording Checklist

### Environment Setup
- [ ] Close unnecessary applications
- [ ] Clear browser tabs (only keep GitHub, localhost:8501)
- [ ] Increase terminal font size (18-20pt)
- [ ] Set Jupyter notebook theme to readable contrast
- [ ] Test microphone levels
- [ ] Run all notebooks once to ensure no errors

### Data Preparation
- [ ] Verify evidence.csv is in ecogrow/ directory
- [ ] Verify impact_math.csv exists
- [ ] Verify carbon_aware_decision.json exists
- [ ] Generate charts from 04_Impact_Analysis_Extended.ipynb beforehand
- [ ] Test dashboard launches successfully

### Script Preparation
- [ ] Print script for reference
- [ ] Rehearse narration (aim for natural, conversational tone)
- [ ] Time each section (adjust pace if over 5 minutes)

---

## 🎥 Recording Steps

### Step 1: Test Recording (5 minutes)
1. Record 30-second test clip
2. Check audio quality (no echo, clear speech)
3. Check screen clarity (text readable)
4. Verify cursor is visible

### Step 2: Record in Segments
**Tip:** Record each section separately, then edit together
- Introduction: 1 take
- Problem Statement: 1 take
- QuickStart Demo: 2-3 takes (may need multiple runs)
- CarbonAware Demo: 2 takes
- Impact Analysis: 2 takes
- Dashboard: 2 takes
- Conclusion: 1 take

### Step 3: Edit
1. Trim silences and pauses
2. Add transitions between sections (1-second fade)
3. Add text overlays for key metrics:
   - "76.5% Energy Reduction"
   - "502,500 People Protected"
   - "4-Month Payback"
4. Add background music (subtle, 20% volume)
5. Add GitHub URL as lower-third graphic

### Step 4: Export
```
Format: MP4
Codec: H.264
Resolution: 1920x1080
Frame Rate: 30 fps
Bitrate: 5-8 Mbps
Audio: AAC 128 kbps
```

---

## 📤 Upload & Submission

### YouTube (Recommended)
1. Create YouTube account (if needed)
2. Upload video as **Unlisted** (searchable via link)
3. Title: "EcoGrow: 76.5% Energy Reduction in Greenhouse AI | HACK4EARTH BUIDL"
4. Description:
```
EcoGrow - Physics-Informed AI for Carbon-Neutral Greenhouse Control

🎯 HACK4EARTH BUIDL Challenge Submission
📊 Track A: 76.5% energy reduction through INT8 quantization
🌍 Track B: 43.2% carbon reduction through carbon-aware scheduling

Results:
- 24,640 tons CO₂ saved annually (at scale)
- 2 million m³ water saved
- 502,500 people protected
- 4-month payback period

🔗 GitHub: https://github.com/Ramesh-Arvind/HACK4EARTH-Green-AI
🔗 DoraHacks: [submission link]

#HACK4EARTH #GreenAI #SustainableAI #ClimateTech
```
5. Tags: HACK4EARTH, Green AI, Sustainable AI, Climate Tech, Greenhouse, Agriculture
6. Thumbnail: Screenshot of dashboard or impact metrics
7. Copy YouTube link

### Alternative: Google Drive / Dropbox
1. Upload video to cloud storage
2. Set sharing to "Anyone with link can view"
3. Copy shareable link

### Add to Submission
1. Update `SUBMISSION.md` with video link
2. Add to README.md under "Video Demo" section
3. Include in DoraHacks BUIDL submission form

---

## ✅ Quality Checklist

### Audio
- [ ] Clear narration (no background noise)
- [ ] Consistent volume levels
- [ ] No long silences (>3 seconds)

### Visual
- [ ] Text is readable (minimum 16pt font)
- [ ] No personal information visible
- [ ] Cursor movements are smooth
- [ ] Transitions are professional

### Content
- [ ] All key metrics mentioned (76.5%, 43.2%, 24,640 tons, etc.)
- [ ] Hardware validation emphasized
- [ ] Open-source nature highlighted
- [ ] GitHub link displayed
- [ ] Duration: 3-5 minutes

### Technical
- [ ] Video plays smoothly (no stuttering)
- [ ] Audio synced with video
- [ ] File size <100 MB (for easy upload)
- [ ] Format compatible (MP4 H.264)

---

## 🎯 Expected Impact

**Before Video:** Openness & Transparency = 17/20
- Documentation: ✅ 5/5
- Code availability: ✅ 5/5
- Reproducibility: ✅ 5/5
- Demo/visualization: ❌ 2/5 (screenshots only)

**After Video:** Openness & Transparency = 20/20
- Documentation: ✅ 5/5
- Code availability: ✅ 5/5
- Reproducibility: ✅ 5/5
- Demo/visualization: ✅ 5/5 (live demonstration)

**BONUS Points:**
- Video demo: +3 points
- Live dashboard: +2 points
- **Total Potential Boost: +5 points**

**New Score:** 95/100 → **100/100** 🏆

---

## 📞 Support

If you encounter issues during recording:
- **Technical issues:** Check GitHub Issues
- **Software problems:** See TROUBLESHOOTING.md
- **Questions:** rameshln.96@gmail.com

Good luck with your video! 🎬🌱
