<a class="kdd-announcement-bar" href="https://kdd2026.kdd.org/call-for-hands-on-tutorials/">
  <span class="kdd-announcement-status">KDD 2026 hands-on tutorial</span>
  <strong>OneEHR</strong>
  <span>Tuesday, August 11, 2026</span>
  <span>1:30 PM - 4:30 PM</span>
  <span>Jeju, Korea</span>
</a>

<div class="landing-hero">
  <div class="landing-hero-copy">
    <p class="landing-eyebrow">EHR AI platform</p>
    <h1>OneEHR</h1>
    <p class="landing-lede">
      Run longitudinal EHR experiments from standardized event tables, a TOML config,
      and one saved run directory.
    </p>
    <p class="landing-body">
      OneEHR covers preprocessing, model training, testing, analysis, and figures for
      conventional ML/DL models plus LLM or agent systems. The same config and artifacts
      are used by the CLI, Python API, and notebooks.
    </p>
    <div class="landing-actions">
      <a class="landing-button landing-button-primary" href="./getting-started/quickstart/">Run the quickstart</a>
      <a class="landing-button landing-button-secondary" href="./getting-started/data-model/">Prepare data</a>
    </div>
  </div>
  <div class="landing-hero-panel">
    <div class="landing-badge-row">
      <span class="landing-badge">Python 3.12+</span>
      <span class="landing-badge">TOML config</span>
      <span class="landing-badge">MIMIC / eICU</span>
      <span class="landing-badge">ICD / CCS / ATC</span>
      <span class="landing-badge">Parquet + JSON</span>
    </div>
    <div class="landing-stats">
      <article class="landing-stat-card">
        <span class="landing-stat-label">Input</span>
        <strong>3-table EHR schema</strong>
        <span class="landing-stat-meta"><code>dynamic.csv</code>, <code>static.csv</code>, <code>label.csv</code></span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Workflow</span>
        <strong>Preprocess to plot</strong>
        <span class="landing-stat-meta">One config, one run directory</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Models</span>
        <strong>42 built in</strong>
        <span class="landing-stat-meta">ML, DL, multimodal, KG, survival</span>
      </article>
      <article class="landing-stat-card">
        <span class="landing-stat-label">Outputs</span>
        <strong>Structured artifacts</strong>
        <span class="landing-stat-meta">Predictions, metrics, analysis, figures</span>
      </article>
    </div>
  </div>
</div>

## Start Here

<div class="entry-grid">
  <article class="entry-card">
    <h3><a href="./getting-started/installation/">Installation</a></h3>
    <p>Set up Python 3.12+, install OneEHR, and verify the CLI.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./getting-started/quickstart/">Quickstart</a></h3>
    <p>Run the bundled TJH example from CSV conversion through analysis and figures.</p>
  </article>
  <article class="entry-card">
    <h3><a href="./getting-started/data-model/">Data Model</a></h3>
    <p>Prepare the dynamic, static, and label CSV files used by every workflow.</p>
  </article>
</div>

## Standard Workflow

<div class="workflow-grid">
  <article class="workflow-step">
    <span class="workflow-step-no">01</span>
    <h3>Preprocess</h3>
    <p>Bin events, encode features, create labels, and save a patient-level split.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">02</span>
    <h3>Train</h3>
    <p>Fit every model listed in the TOML config against the saved artifacts.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">03</span>
    <h3>Test</h3>
    <p>Evaluate trained models and configured systems on the held-out test split.</p>
  </article>
  <article class="workflow-step">
    <span class="workflow-step-no">04</span>
    <h3>Analyze</h3>
    <p>Write comparison, feature importance, fairness, calibration, statistical test, and missing-data outputs.</p>
  </article>
</div>

```bash
oneehr preprocess --config experiment.toml
oneehr train      --config experiment.toml
oneehr test       --config experiment.toml
oneehr analyze    --config experiment.toml
oneehr plot       --config experiment.toml
```

## What You Can Use

<div class="feature-grid">
  <article class="feature-card">
    <p class="feature-kicker">Data</p>
    <h3>Standard CSV inputs</h3>
    <p>Prepare longitudinal events once, then reuse the same tables for patient-level and time-level tasks.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Models</p>
    <h3>Configured model runs</h3>
    <p>Select tabular, deep learning, irregular-time, multimodal, KG-enhanced, or survival models with <code>[[models]]</code> blocks.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Systems</p>
    <h3>LLM and agent evaluation</h3>
    <p>Add <code>[[systems]]</code> entries to write predictions into the same test artifact as model outputs.</p>
  </article>
  <article class="feature-card">
    <p class="feature-kicker">Analysis</p>
    <h3>Machine-readable outputs</h3>
    <p>Use Parquet predictions and JSON analysis files for notebooks, reports, and downstream tooling.</p>
  </article>
</div>

## Common Next Pages

- [Core Workflows](guide/core-workflows.md) explains each CLI stage and its outputs.
- [Configuration Reference](reference/configuration.md) lists TOML fields and defaults.
- [Models Reference](reference/models.md) lists all 42 model config names and parameters.
- [Artifacts Reference](reference/artifacts.md) documents the on-disk run contract.
- [Dataset Converters](reference/datasets.md) covers MIMIC-III, MIMIC-IV, and eICU conversion.
- [Medical Codes](reference/medcode.md) covers ICD, CCS, ATC, and code mapping helpers.

## Tutorial Tutors

<div class="tutor-grid">
  <a class="tutor-card" href="https://yhzhu99.github.io/">
    <img class="tutor-avatar" src="images/team/yinghao-zhu.jpg" alt="Yinghao Zhu">
    <span class="tutor-name">Yinghao Zhu</span>
    <span class="tutor-affiliation">Peking University; University of Hong Kong</span>
  </a>
  <a class="tutor-card" href="https://fieldry.github.io/">
    <img class="tutor-avatar" src="images/team/zixiang-wang.jpg" alt="Zixiang Wang">
    <span class="tutor-name">Zixiang Wang</span>
    <span class="tutor-affiliation">Peking University</span>
  </a>
  <a class="tutor-card" href="https://openreview.net/profile?id=~Lei_Gu5">
    <img class="tutor-avatar" src="images/team/lei-gu.jpg" alt="Lei Gu">
    <span class="tutor-name">Lei Gu</span>
    <span class="tutor-affiliation">Peking University</span>
  </a>
  <a class="tutor-card" href="https://openreview.net/profile?id=~Dehao_Sui1">
    <img class="tutor-avatar" src="images/team/dehao-sui.jpg" alt="Dehao Sui">
    <span class="tutor-name">Dehao Sui</span>
    <span class="tutor-affiliation">Peking University</span>
  </a>
  <a class="tutor-card" href="http://scholar.pku.edu.cn/wangyasha/">
    <img class="tutor-avatar" src="images/team/yasha-wang.jpg" alt="Yasha Wang">
    <span class="tutor-name">Yasha Wang</span>
    <span class="tutor-affiliation">Peking University</span>
  </a>
  <a class="tutor-card" href="https://www.ed.ac.uk/profile/ewen-harrison">
    <img class="tutor-avatar" src="images/team/ewen-harrison.jpg" alt="Ewen M. Harrison">
    <span class="tutor-name">Ewen M. Harrison</span>
    <span class="tutor-affiliation">University of Edinburgh</span>
  </a>
  <a class="tutor-card" href="https://futianfan.github.io/">
    <img class="tutor-avatar" src="images/team/tianfan-fu.jpg" alt="Tianfan Fu">
    <span class="tutor-name">Tianfan Fu</span>
    <span class="tutor-affiliation">Nanjing University</span>
  </a>
  <a class="tutor-card" href="https://aboutme.vixerunt.org/">
    <img class="tutor-avatar" src="images/team/junyi-gao.jpg" alt="Junyi Gao">
    <span class="tutor-name">Junyi Gao</span>
    <span class="tutor-affiliation">University of Edinburgh; Health Data Research UK</span>
  </a>
  <a class="tutor-card" href="https://yulequan.github.io/">
    <img class="tutor-avatar" src="images/team/lequan-yu.jpg" alt="Lequan Yu">
    <span class="tutor-name">Lequan Yu</span>
    <span class="tutor-affiliation">University of Hong Kong</span>
  </a>
  <a class="tutor-card" href="https://medx-pku.com/malt">
    <img class="tutor-avatar" src="images/team/liantao-ma.jpg" alt="Liantao Ma">
    <span class="tutor-name">Liantao Ma</span>
    <span class="tutor-affiliation">Peking University</span>
  </a>
</div>
